"""Model-input preparation and benchmark statistics independent of ROS 2."""

import shutil
import statistics
import subprocess
import platform
from pathlib import Path

import numpy as np


def _input_layout(shape):
    if len(shape) != 4 or shape[0] not in (1, "batch", None):
        raise ValueError(f"Expected a batch-one rank-4 model input, got {shape}")
    if shape[1] in (1, 3, 4):
        return "NCHW", int(shape[1])
    if shape[3] in (1, 3, 4):
        return "NHWC", int(shape[3])
    raise ValueError(f"Cannot infer image channel layout from model input shape {shape}")


def preprocess_image(image, model_shape, scale=1.0 / 255.0, mean=None, std=None):
    """Resize an RGB image and return a normalized float32 tensor for a rank-4 ONNX input."""
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


def timing_stats(samples_ns):
    """Return mean, min, max and sample standard deviation in nanoseconds."""
    samples = np.asarray(samples_ns, dtype=np.float64)
    if samples.size == 0:
        raise ValueError("At least one timing sample is required")
    return {
        "avg": float(np.mean(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "std": float(statistics.stdev(samples)) if samples.size > 1 else 0.0,
    }


def sample_system(interval=0.1):
    """Capture host memory and CPU utilization outside timed inference calls."""
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
    """Capture NVIDIA GPU name, memory use and utilization when nvidia-smi exists."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = (
        "--query-gpu=index,name,driver_version,memory.total,memory.used,"
        "memory.free,utilization.gpu --format=csv,noheader,nounits"
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
            parse_number = lambda value: None if value.upper() == "N/A" else int(value)
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_total_mb": parse_number(fields[3]),
                    "memory_used_mb": parse_number(fields[4]),
                    "memory_free_mb": parse_number(fields[5]),
                    "util_percent": parse_number(fields[6]),
                }
            )
        except ValueError:
            continue
    return gpus or None