"""Model-input preparation and benchmark statistics independent of ROS 2."""

import json
import math
import os
import platform
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CAMERA_DATA_TIMEOUT_EXIT_CODE = 75
PREPROCESSING_MANIFEST_VERSION = 1
DROPPED_FRAME_TOLERANCE_FRACTION = 0.005


def resolve_onnx_providers(mode, available_providers, configured_providers=None):
    """Choose ONNX Runtime providers for automatic, CPU-only, or CUDA-only runs."""
    if mode == "cpu":
        required_provider = "CPUExecutionProvider"
        requested = [required_provider]
    elif mode == "cuda":
        required_provider = "CUDAExecutionProvider"
        requested = [required_provider]
    elif mode == "auto":
        requested = list(
            configured_providers
            if configured_providers is not None
            else ("CUDAExecutionProvider", "CPUExecutionProvider")
        )
        required_provider = None
    else:
        raise ValueError("ONNX execution mode must be 'auto', 'cpu', or 'cuda'")

    available = set(available_providers)
    if required_provider is not None and required_provider not in available:
        raise RuntimeError(
            f"Requested ONNX Runtime provider {required_provider} is unavailable; "
            f"available providers: {sorted(available)}"
        )
    providers = [provider for provider in requested if provider in available]
    if not providers:
        raise RuntimeError(
            f"None of the requested ONNX Runtime providers are available: "
            f"{requested}; available: {sorted(available)}"
        )
    return providers


def _input_layout(shape):
    if len(shape) != 4 or shape[0] not in (1, "batch", None):
        raise ValueError(f"Expected a batch-one rank-4 model input, got {shape}")
    if shape[1] in (1, 3, 4):
        return "NCHW", int(shape[1])
    if shape[3] in (1, 3, 4):
        return "NHWC", int(shape[3])
    raise ValueError(f"Cannot infer image channel layout from model input shape {shape}")


def load_preprocessing_manifest(path):
    """Load and validate a model-specific preprocessing manifest."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read preprocessing manifest {path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("version") != PREPROCESSING_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported preprocessing manifest version in {path}; "
            f"expected {PREPROCESSING_MANIFEST_VERSION}"
        )
    operations = manifest.get("operations")
    if not isinstance(operations, list):
        raise ValueError(f"Preprocessing manifest {path} must contain an operations list")
    supported_operations = {"resize", "center_crop", "grayscale", "to_tensor", "normalize"}
    if not operations:
        raise ValueError(f"Preprocessing manifest {path} must not have an empty operations list")
    seen_to_tensor = False
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or operation.get("type") not in supported_operations:
            raise ValueError(f"Unsupported preprocessing operation at index {index} in {path}")
        operation_type = operation["type"]
        if operation_type in {"resize", "center_crop"}:
            size = operation.get("size")
            valid_size = (
                isinstance(size, int) and size > 0
            ) or (
                isinstance(size, list)
                and len(size) == 2
                and all(isinstance(dimension, int) and dimension > 0 for dimension in size)
            )
            if not valid_size:
                raise ValueError(f"Invalid {operation_type} size at index {index} in {path}")
            if operation_type == "resize":
                if operation.get("mode") not in {"shorter", "exact"}:
                    raise ValueError(f"Invalid resize mode at index {index} in {path}")
                if operation.get("interpolation", "bilinear") not in {
                    "nearest", "bilinear", "bicubic", "lanczos"
                }:
                    raise ValueError(f"Invalid resize interpolation at index {index} in {path}")
        elif operation_type == "to_tensor":
            if seen_to_tensor:
                raise ValueError(f"Manifest {path} has multiple to_tensor operations")
            scale = operation.get("scale", 1.0 / 255.0)
            if not isinstance(scale, (int, float)) or not math.isfinite(scale):
                raise ValueError(f"Invalid tensor scale at index {index} in {path}")
            seen_to_tensor = True
        elif operation_type == "normalize":
            mean = operation.get("mean")
            std = operation.get("std")
            if (
                not seen_to_tensor
                or not isinstance(mean, list)
                or not isinstance(std, list)
                or not mean
                or len(mean) != len(std)
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in mean + std)
                or any(value == 0 for value in std)
            ):
                raise ValueError(f"Invalid normalization at index {index} in {path}")
    if not seen_to_tensor:
        raise ValueError(f"Preprocessing manifest {path} must contain a to_tensor operation")
    return manifest


def _preprocess_with_manifest(image, model_shape, operations):
    from PIL import Image, ImageOps

    layout, channels = _input_layout(model_shape)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got shape {image.shape}")
    pil_image = Image.fromarray(image[..., :3].astype(np.uint8, copy=False))
    tensor = None

    for operation in operations:
        operation_type = operation["type"]
        if operation_type == "grayscale":
            if tensor is not None:
                raise ValueError("grayscale must appear before to_tensor in the manifest")
            pil_image = pil_image.convert("L")
        elif operation_type == "resize":
            if tensor is not None:
                raise ValueError("resize must appear before to_tensor in the manifest")
            size = operation.get("size")
            interpolation = operation.get("interpolation", "bilinear").lower()
            resampling = {
                "nearest": Image.Resampling.NEAREST,
                "bilinear": Image.Resampling.BILINEAR,
                "bicubic": Image.Resampling.BICUBIC,
                "lanczos": Image.Resampling.LANCZOS,
            }.get(interpolation)
            if resampling is None:
                raise ValueError(f"Unsupported resize interpolation {interpolation!r}")
            if isinstance(size, int) and operation.get("mode") == "shorter":
                width, height = pil_image.size
                if width <= height:
                    output_size = (size, int(size * height / width))
                else:
                    output_size = (int(size * width / height), size)
            elif isinstance(size, int):
                output_size = (size, size)
            elif isinstance(size, list) and len(size) == 2:
                output_size = (int(size[1]), int(size[0]))
            else:
                raise ValueError(f"Invalid resize size in manifest: {size!r}")
            pil_image = pil_image.resize(output_size, resample=resampling)
        elif operation_type == "center_crop":
            if tensor is not None:
                raise ValueError("center_crop must appear before to_tensor in the manifest")
            size = operation.get("size")
            if isinstance(size, int):
                crop_height = crop_width = size
            elif isinstance(size, list) and len(size) == 2:
                crop_height, crop_width = map(int, size)
            else:
                raise ValueError(f"Invalid center-crop size in manifest: {size!r}")
            width, height = pil_image.size
            pad_width = max(crop_width - width, 0)
            pad_height = max(crop_height - height, 0)
            if pad_width or pad_height:
                left_pad = pad_width // 2
                top_pad = pad_height // 2
                pil_image = ImageOps.expand(
                    pil_image,
                    border=(
                        left_pad,
                        top_pad,
                        pad_width - left_pad,
                        pad_height - top_pad,
                    ),
                    fill=0,
                )
                width, height = pil_image.size
            left = int(round((width - crop_width) / 2.0))
            top = int(round((height - crop_height) / 2.0))
            pil_image = pil_image.crop((left, top, left + crop_width, top + crop_height))
        elif operation_type == "to_tensor":
            if tensor is not None:
                raise ValueError("Manifest contains more than one to_tensor operation")
            scale = float(operation.get("scale", 1.0 / 255.0))
            if not math.isfinite(scale):
                raise ValueError("Manifest tensor scale must be finite")
            pixels = np.asarray(pil_image)
            if pixels.ndim == 2:
                pixels = pixels[..., None]
            tensor = pixels.astype(np.float32) * scale
        elif operation_type == "normalize":
            if tensor is None:
                raise ValueError("normalize must appear after to_tensor in the manifest")
            mean = np.asarray(operation.get("mean"), dtype=np.float32)
            std = np.asarray(operation.get("std"), dtype=np.float32)
            if mean.ndim != 1 or std.ndim != 1 or len(mean) != channels or len(std) != channels:
                raise ValueError(f"Manifest normalization must define {channels} channels")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std == 0):
                raise ValueError("Manifest normalization values must be finite with non-zero std")
            tensor = (tensor - mean.reshape(1, 1, channels)) / std.reshape(1, 1, channels)

    if tensor is None:
        raise ValueError("Preprocessing manifest must include a to_tensor operation")
    if tensor.shape[-1] != channels:
        raise ValueError(
            f"Preprocessed image has {tensor.shape[-1]} channels but model expects {channels}"
        )
    if layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))
    result = np.expand_dims(tensor, axis=0).astype(np.float32, copy=False)
    expected_shape = tuple(
        int(size) if isinstance(size, (int, np.integer)) else result.shape[index]
        for index, size in enumerate(model_shape)
    )
    if result.shape != expected_shape:
        raise ValueError(
            f"Preprocessed tensor shape {result.shape} does not match model input {expected_shape}"
        )
    return result


def preprocess_image(
    image, model_shape, scale=1.0 / 255.0, mean=None, std=None, operations=None
):
    """Prepare an image using a manifest or the legacy nearest-resize parameters."""
    if operations is not None:
        return _preprocess_with_manifest(image, model_shape, operations)

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got shape {image.shape}")

    layout, channels = _input_layout(model_shape)
    if layout == "NCHW":
        height, width = model_shape[2:4]
    else:
        height, width = model_shape[1:3]
    height = int(height) if isinstance(height, int) else image.shape[0]
    width = int(width) if isinstance(width, int) else image.shape[1]
    if height < 1 or width < 1:
        raise ValueError(f"Invalid model image dimensions: {height}x{width}")

    y_indices = np.linspace(0, image.shape[0] - 1, height).astype(np.intp)
    x_indices = np.linspace(0, image.shape[1] - 1, width).astype(np.intp)
    resized = image[y_indices[:, None], x_indices[None, :], :3]
    if channels == 1:
        resized = np.rint(
            resized[..., 0] * 0.299 + resized[..., 1] * 0.587 + resized[..., 2] * 0.114
        )[..., None]
    elif channels == 4:
        resized = np.concatenate(
            (resized, np.full((*resized.shape[:2], 1), 255, dtype=resized.dtype)), axis=2
        )

    tensor = resized.astype(np.float32) * float(scale)
    if mean:
        if len(mean) != channels:
            raise ValueError(f"input_mean must have {channels} values")
        tensor -= np.asarray(mean, dtype=np.float32).reshape(1, 1, channels)
    if std:
        if len(std) != channels or any(float(value) == 0 for value in std):
            raise ValueError(f"input_std must have {channels} non-zero values")
        tensor /= np.asarray(std, dtype=np.float32).reshape(1, 1, channels)

    if layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32, copy=False)


def l2_normalize(output):
    """Return a float32 output normalized to unit L2 norm, preserving zero vectors."""
    output = np.asarray(output, dtype=np.float32)
    norm = np.linalg.norm(output)
    if norm == 0:
        return output
    return output / norm


def process_model_output(output, normalization="none"):
    """Apply the explicitly selected output transform without changing raw outputs."""
    if normalization == "none":
        return np.asarray(output, dtype=np.float32)
    if normalization == "l2":
        return l2_normalize(output)
    raise ValueError(f"Unsupported output normalization {normalization!r}")


def timing_stats(samples_ns, deadline_ns=None):
    """Return distribution statistics and all measured nanosecond samples.

    Standard deviation is computed as the population standard deviation
    (dividing by N, ddof=0), matching nn-lite's convention (unlike sample
    standard deviation with Bessel's correction N-1).
    """
    samples = np.asarray(samples_ns, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("At least one timing sample is required")
    stats = {
        "avg": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "std": float(np.std(samples)),
        "samples": samples.tolist(),
    }
    if deadline_ns is not None:
        stats["deadline_misses"] = int(np.sum(samples > deadline_ns))
    return stats


def synthetic_rgb_frame(width, height, frame_index=0):
    """Create a deterministic RGB test pattern at the requested camera dimensions."""
    if width < 1 or height < 1:
        raise ValueError(f"Frame dimensions must be positive, got {width}x{height}")
    x = np.arange(width, dtype=np.uint32)[None, :]
    y = np.arange(height, dtype=np.uint32)[:, None]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[..., 0] = (
        (x * 255 // max(width - 1, 1) + frame_index * 3) % 256
    ).astype(np.uint8)
    frame[..., 1] = (y * 255 // max(height - 1, 1)).astype(np.uint8)
    frame[..., 2] = (((x // 32 + y // 32 + frame_index) % 2) * 255).astype(np.uint8)
    return frame


def message_stamp_ns(message):
    """Return a ROS message header stamp as nanoseconds, or zero when unset."""
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return 0
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def header_to_publish_latency_ns(source_stamp_ns, callback_ros_ns, published_ros_ns):
    """Return header-stamp-to-publish latency, rejecting timestamps from inconsistent clocks."""
    if source_stamp_ns <= 0:
        return None
    if source_stamp_ns > callback_ros_ns or source_stamp_ns > published_ros_ns:
        raise ValueError(
            "Image header timestamp is later than callback or publish ROS time; "
            "clock domains may be inconsistent"
        )
    return published_ros_ns - source_stamp_ns


def dropped_frames_between(previous_stamp_ns, current_stamp_ns, frame_rate_hz):
    """Estimate missing camera frames from the elapsed time between source image stamps."""
    if current_stamp_ns <= previous_stamp_ns:
        raise ValueError("Image header timestamps are not increasing")
    frame_period_ns = 1_000_000_000 / frame_rate_hz
    expected_intervals = max(
        1, int(round((current_stamp_ns - previous_stamp_ns) / frame_period_ns))
    )
    return expected_intervals - 1


def dropped_frame_fraction(dropped_frames, received_frames):
    """Estimate the missing-frame fraction over received plus inferred frames."""
    if dropped_frames < 0 or received_frames < 0:
        raise ValueError("Frame counts cannot be negative")
    total_frames = dropped_frames + received_frames
    return dropped_frames / total_frames if total_frames else 0.0


def dropped_frames_within_tolerance(dropped_frames, received_frames):
    """Allow only small timestamp-derived loss while separately checking input rate."""
    return (
        dropped_frame_fraction(dropped_frames, received_frames)
        <= DROPPED_FRAME_TOLERANCE_FRACTION
    )


def sample_temperatures():
    """Capture CPU, GPU, and system temperatures in Celsius when available."""
    temps = {}
    try:
        import psutil

        if hasattr(psutil, "sensors_temperatures"):
            sensor_data = psutil.sensors_temperatures()
            if sensor_data:
                for name, entries in sensor_data.items():
                    temps[name] = [
                        {
                            "label": e.label or name,
                            "current_c": e.current,
                            "high_c": e.high,
                            "critical_c": e.critical,
                        }
                        for e in entries
                    ]
    except Exception:
        pass
    return temps or None


def sample_cpu_frequencies():
    """Capture CPU clock frequencies and scaling governors when available on Linux."""
    info = {}
    try:
        import psutil

        freq = psutil.cpu_freq()
        if freq:
            current = freq.current
            if (
                isinstance(current, (int, float))
                and isinstance(freq.max, (int, float))
                and freq.max >= 1000
                and 0 < current < 100
            ):
                current *= 1000
            info["cur_freq_mhz"] = current
            info["min_freq_mhz"] = freq.min
            info["max_freq_mhz"] = freq.max
    except Exception:
        pass

    try:
        governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        if governor_path.is_file():
            info["scaling_governor"] = governor_path.read_text().strip()
    except OSError:
        pass
    return info or None


class CpuTelemetryMonitor:
    """Sample per-core CPU utilization and frequency during a measured window."""

    def __init__(self, sampling_interval_seconds=1.0, include_machine_telemetry=False):
        if not math.isfinite(sampling_interval_seconds) or sampling_interval_seconds <= 0:
            raise ValueError("sampling_interval_seconds must be finite and positive")
        self.sampling_interval_seconds = float(sampling_interval_seconds)
        self.include_machine_telemetry = include_machine_telemetry
        self._stop_event = threading.Event()
        self._samples_lock = threading.Lock()
        self._samples = []
        self._sampling_errors = []
        self._started_at = None
        self._process = None
        self._thread = None
        self._psutil = None

    @property
    def started(self):
        return self._thread is not None

    def start(self):
        if self.started:
            raise RuntimeError("CPU telemetry monitor has already started")
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        psutil.cpu_percent(interval=None, percpu=True)
        self._process.cpu_percent(interval=None)
        self._started_at = time.monotonic()
        self._capture_sample(include_utilization=False)
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="cpu-telemetry-monitor",
            daemon=True,
        )
        self._thread.start()

    def _capture_sample(self, include_utilization):
        psutil = self._psutil
        frequencies = psutil.cpu_freq(percpu=True) or []
        cpu_utilization = (
            psutil.cpu_percent(interval=None, percpu=True)
            if include_utilization
            else []
        )
        process_utilization = (
            self._process.cpu_percent(interval=None)
            if include_utilization
            else None
        )
        try:
            context_switches = self._process.num_ctx_switches()
            process_context_switches = {
                "voluntary": context_switches.voluntary,
                "involuntary": context_switches.involuntary,
            }
        except (self._psutil.Error, OSError, AttributeError):
            process_context_switches = None
        per_core = []
        for index, frequency in enumerate(frequencies):
            current = frequency.current
            maximum = frequency.max
            if (
                isinstance(current, (int, float))
                and isinstance(maximum, (int, float))
                and maximum >= 1000
                and 0 < current < 100
            ):
                current *= 1000
            per_core.append(
                {
                    "cpu": index,
                    "utilization_percent": (
                        cpu_utilization[index]
                        if index < len(cpu_utilization)
                        else None
                    ),
                    "frequency_mhz": current,
                    "min_frequency_mhz": frequency.min,
                    "max_frequency_mhz": maximum,
                }
            )
        sample = {
            "elapsed_seconds": time.monotonic() - self._started_at,
            "process_cpu_percent": process_utilization,
            "process_context_switches": process_context_switches,
            "per_core": per_core,
        }
        if self.include_machine_telemetry and include_utilization:
            sample["temperatures"] = sample_temperatures()
            sample["gpus"] = sample_gpus()
        with self._samples_lock:
            self._samples.append(sample)

    def _sample_loop(self):
        while not self._stop_event.wait(self.sampling_interval_seconds):
            try:
                self._capture_sample(include_utilization=True)
            except (self._psutil.Error, OSError) as error:
                with self._samples_lock:
                    self._sampling_errors.append(
                        f"{type(error).__name__}: {error}"
                    )

    def stop_and_report(self):
        if self._thread is None:
            return {
                "sampling_interval_seconds": self.sampling_interval_seconds,
                "sample_count": 0,
                "samples": [],
                "sampling_errors": [],
            }
        ended_at = time.monotonic()
        self._stop_event.set()
        self._thread.join(timeout=self.sampling_interval_seconds + 3.0)
        if self._thread.is_alive():
            raise RuntimeError("CPU telemetry monitor did not stop")
        with self._samples_lock:
            samples = list(self._samples)
            sampling_errors = list(self._sampling_errors)
        return {
            "sampling_interval_seconds": self.sampling_interval_seconds,
            "sample_count": len(samples),
            "measurement_window_seconds": (
                ended_at - self._started_at
            ),
            "frequency_samples_are_diagnostic_only": True,
            "samples": samples,
            "sampling_errors": sampling_errors,
        }


def measurement_condition_warnings(cpu_frequencies, temperatures):
    """Report CPU conditions that make inference timings difficult to compare."""
    warnings = []
    governor = (cpu_frequencies or {}).get("scaling_governor")
    if governor == "powersave":
        warnings.append("CPU scaling governor is powersave")

    for sensor, entries in (temperatures or {}).items():
        for entry in entries:
            current_c = entry.get("current_c")
            if isinstance(current_c, (int, float)) and current_c >= 85:
                warnings.append(
                    f"{entry.get('label') or sensor} temperature is at least 85 C"
                )
    return warnings


def assess_gpu_load(gpus, utilization_limit=20.0, memory_limit_percent=20.0):
    """Assess sampled GPU utilization and memory against benchmark limits."""
    thresholds = {
        "utilization_percent": utilization_limit,
        "memory_used_percent": memory_limit_percent,
    }
    if not gpus:
        return {
            "available": False,
            "busy": False,
            "thresholds": thresholds,
            "devices": [],
            "warnings": ["GPU usage could not be sampled (nvidia-smi unavailable or failed)"],
        }

    devices = []
    warnings = []
    for gpu in gpus:
        memory_total = gpu.get("memory_total_mb")
        memory_used = gpu.get("memory_used_mb")
        memory_used_percent = (
            100.0 * memory_used / memory_total
            if memory_total and memory_used is not None
            else None
        )
        device = {
            **gpu,
            "memory_used_percent": memory_used_percent,
            "busy": False,
        }
        if (
            gpu.get("util_percent") is not None
            and gpu["util_percent"] > utilization_limit
        ):
            device["busy"] = True
            warnings.append(
                f'GPU {gpu.get("index", "?")} utilization is {gpu["util_percent"]}% '
                f"(limit {utilization_limit:g}%)"
            )
        if (
            memory_used_percent is not None
            and memory_used_percent > memory_limit_percent
        ):
            device["busy"] = True
            warnings.append(
                f'GPU {gpu.get("index", "?")} memory use is {memory_used_percent:.1f}% '
                f"({memory_used} of {memory_total} MB; limit {memory_limit_percent:g}%)"
            )
        devices.append(device)

    return {
        "available": True,
        "busy": any(device["busy"] for device in devices),
        "thresholds": thresholds,
        "devices": devices,
        "warnings": warnings,
    }


def summarize_onnxruntime_profile(path):
    """Summarize node execution providers from an ONNX Runtime JSON trace."""
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    event_counts = Counter()
    unique_nodes = defaultdict(set)
    operator_counts = defaultdict(Counter)
    for event in events:
        if event.get("cat") != "Node":
            continue
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        node_name = event.get("name")
        if not node_name:
            continue
        event_counts[provider] += 1
        unique_nodes[provider].add(node_name)
        operator = args.get("op_name")
        if operator:
            operator_counts[provider][operator] += 1

    if not event_counts:
        raise ValueError(f"ONNX Runtime profile contains no provider-tagged node events: {path}")

    return {
        "trace_path": str(Path(path)),
        "node_execution_event_counts": dict(event_counts),
        "unique_node_counts_by_provider": {
            provider: len(names) for provider, names in unique_nodes.items()
        },
        "operator_event_counts_by_provider": {
            provider: dict(counts) for provider, counts in operator_counts.items()
        },
    }


def set_cpu_affinity(cpus):
    """Pin the current process to the given CPU core(s)."""
    if hasattr(os, "sched_setaffinity"):
        try:
            if isinstance(cpus, int):
                cpus = [cpus]
            os.sched_setaffinity(0, set(cpus))
            return True
        except (OSError, ValueError):
            return False
    return False


def get_cpu_affinity():
    """Return the CPU affinity mask of the current process."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return sorted(os.sched_getaffinity(0))
        except OSError:
            return None
    return None


def sample_system(interval=0.1):
    """Capture host memory, CPU utilization, frequencies and temperatures."""
    try:
        import psutil
    except ImportError:
        return None

    memory = psutil.virtual_memory()
    process = psutil.Process()
    try:
        process_rss = process.memory_info().rss
        process_cpu = process.cpu_percent(interval=None)
    except psutil.Error:
        process_rss = None
        process_cpu = None
    return {
        "total_ram_bytes": memory.total,
        "free_ram_bytes": memory.free,
        "available_ram_bytes": memory.available,
        "used_ram_bytes": memory.used,
        "cached_ram_bytes": getattr(memory, "cached", 0),
        "ram_percent": memory.percent,
        "cpu_percent": psutil.cpu_percent(interval=interval),
        "process_cpu_percent": process_cpu,
        "process_rss_bytes": process_rss,
        "cpu_frequencies": sample_cpu_frequencies(),
        "temperatures": sample_temperatures(),
    }


def device_name():
    """Return the machine vendor/model when Linux exposes DMI identifiers."""
    if platform.system() == "Linux":
        try:
            vendor = Path("/sys/class/dmi/id/sys_vendor").read_text().strip()
            product = Path("/sys/class/dmi/id/product_name").read_text().strip()
            if vendor or product:
                return f"{vendor} {product}".strip()
        except OSError:
            pass
    return platform.node() or "unknown_device"


def sample_gpus():
    """Capture NVIDIA GPU name, memory use, utilization and temperature when nvidia-smi exists."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = (
        "--query-gpu=index,name,driver_version,memory.total,memory.used,"
        "memory.free,utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.current.memory --format=csv,noheader,nounits"
    )
    try:
        result = subprocess.run(
            [executable, *query.split()],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            parse_number = lambda value: None if value.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]"} else int(value)
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_total_mb": parse_number(fields[3]),
                    "memory_used_mb": parse_number(fields[4]),
                    "memory_free_mb": parse_number(fields[5]),
                    "util_percent": parse_number(fields[6]),
                    "temperature_c": parse_number(fields[7]) if len(fields) > 7 else None,
                    "graphics_clock_mhz": parse_number(fields[8]) if len(fields) > 8 else None,
                    "memory_clock_mhz": parse_number(fields[9]) if len(fields) > 9 else None,
                }
            )
        except ValueError:
            continue
    return gpus or None