"""ROS 2 node for running an ONNX model on a robot's image stream."""

import json
import math
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

from ab.bot.frame_trace import frame_key, tag_output, write_observation
from ab.bot.runtime import (
    CAMERA_DATA_TIMEOUT_EXIT_CODE,
    CpuTelemetryMonitor,
    DROPPED_FRAME_TOLERANCE_FRACTION,
    assess_gpu_load,
    device_name,
    dropped_frame_fraction,
    dropped_frames_between,
    dropped_frames_within_tolerance,
    get_cpu_affinity,
    header_to_publish_latency_ns,
    load_preprocessing_manifest,
    message_stamp_ns,
    measurement_condition_warnings,
    preprocess_image,
    process_model_output,
    resolve_onnx_providers,
    sample_cpu_frequencies,
    sample_gpus,
    sample_temperatures,
    set_cpu_affinity,
    summarize_onnxruntime_profile,
    timing_stats,
)


class InferenceNode(Node):
    def __init__(self):
        super().__init__("nn_bot_inference")
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.declare_parameter("model_path", "")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("camera_data_timeout", 90.0)
        self.declare_parameter("output_topic", "/nn_bot/output")
        self.declare_parameter("trace_output_frames", False)
        self.trace_output_frames = bool(self.get_parameter("trace_output_frames").value)
        self.declare_parameter("inference_trace_path", "")
        self.inference_trace_path = self.get_parameter("inference_trace_path").value
        self.inference_trace = None
        self.declare_parameter("preprocessing_manifest", "")
        self.declare_parameter("output_normalization", "none")
        self.declare_parameter("results_dir", "stats")
        self.declare_parameter("dataset", "custom")
        self.declare_parameter("synthetic_input", False)
        self.declare_parameter("camera_rate_hz", 0.0)
        self.declare_parameter("warmup_runs", 10)
        self.declare_parameter("warmup_max_runs", 0)
        self.declare_parameter("min_measurement_seconds", 0.0)
        self.declare_parameter("camera_cpu_affinity_cpus", "")
        self.declare_parameter("iterations", 500)
        self.declare_parameter("deadline_ms", 33.0)
        self.declare_parameter("cpu_affinity", -1)
        self.declare_parameter("cpu_affinity_cpus", "")
        self.declare_parameter("onnx_num_threads", 0)
        self.declare_parameter("allow_spinning", False)
        self.declare_parameter("cuda_io_binding", False)
        self.declare_parameter("measurement_mode", "frame")
        self.declare_parameter("warmup_convergence_window", 0)
        self.declare_parameter("warmup_convergence_tolerance_percent", 2.0)
        self.declare_parameter("realtime_priority", 0)
        self.declare_parameter("lock_memory", False)
        self.declare_parameter("real_time_factor", 1.0)
        self.declare_parameter("emulator", False)
        self.declare_parameter("gpu_utilization_limit", 20.0)
        self.declare_parameter("gpu_memory_limit_percent", 20.0)
        self.declare_parameter("enable_onnx_profiling", False)
        self.declare_parameter("onnx_execution_mode", "auto")
        self.declare_parameter("startup_drop_frames", 10)
        self.declare_parameter("ready_topic", "/nn_bot/ready")
        self.declare_parameter("input_scale", 1.0 / 255.0)
        self.declare_parameter("input_mean", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("input_std", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter(
            "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.declare_parameter("exit_after_benchmark", False)

        cpu_affinity = int(self.get_parameter("cpu_affinity").value)
        if cpu_affinity >= 0:
            if set_cpu_affinity(cpu_affinity):
                self.get_logger().info(f"Pinned process to CPU core {cpu_affinity}")
            else:
                self.get_logger().warning(
                    f"Could not pin process to CPU core {cpu_affinity}"
                )
        else:
            affinity_cpus = [
                int(cpu.strip())
                for cpu in str(
                    self.get_parameter("cpu_affinity_cpus").value
                ).split(",")
                if cpu.strip()
            ]
            if affinity_cpus:
                if set_cpu_affinity(affinity_cpus):
                    self.get_logger().info(
                        f"Pinned process to CPU cores {affinity_cpus}"
                    )
                else:
                    self.get_logger().warning(
                        f"Could not pin process to CPU cores {affinity_cpus}"
                    )

        realtime_priority = int(self.get_parameter("realtime_priority").value)
        scheduling_status = "not_requested"
        scheduling_error = None
        if realtime_priority > 0:
            try:
                os.sched_setscheduler(
                    0,
                    os.SCHED_FIFO,
                    os.sched_param(realtime_priority),
                )
                scheduling_status = "SCHED_FIFO"
            except (AttributeError, OSError, ValueError) as error:
                scheduling_status = "requested_but_unavailable"
                scheduling_error = f"{type(error).__name__}: {error}"
        if bool(self.get_parameter("lock_memory").value):
            try:
                import resource
                _, hard_memlock_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                resource.setrlimit(
                    resource.RLIMIT_MEMLOCK,
                    (hard_memlock_limit, hard_memlock_limit),
                )
            except (AttributeError, OSError, ValueError) as error:
                scheduling_status = "partial"
                scheduling_error = (
                    f"{scheduling_error}; " if scheduling_error else ""
                ) + f"mlockall unavailable: {type(error).__name__}: {error}"
            else:
                try:
                    import ctypes
                    libc = ctypes.CDLL(None, use_errno=True)
                    if libc.mlockall(1 | 2) != 0:
                        error_number = ctypes.get_errno()
                        raise OSError(
                            error_number,
                            os.strerror(error_number),
                        )
                except (AttributeError, OSError) as error:
                    scheduling_error = (
                        f"{scheduling_error}; " if scheduling_error else ""
                    ) + f"mlockall unavailable: {type(error).__name__}: {error}"

        model_path = Path(self.get_parameter("model_path").value).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        configured_manifest_path = str(
            self.get_parameter("preprocessing_manifest").value
        ).strip()
        if configured_manifest_path:
            manifest_path = Path(configured_manifest_path).expanduser()
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Preprocessing manifest not found: {manifest_path}"
                )
        else:
            manifest_path = model_path.with_suffix(".preprocess.json")
        self.preprocessing_manifest = (
            load_preprocessing_manifest(manifest_path)
            if manifest_path.is_file()
            else None
        )
        self.preprocessing_manifest_path = (
            manifest_path if self.preprocessing_manifest is not None else None
        )
        self.results_dir = Path(self.get_parameter("results_dir").value).expanduser()
        if (
            self.preprocessing_manifest is not None
            and self.preprocessing_manifest.get("model") not in (None, model_path.stem)
        ):
            raise ValueError(
                f"Preprocessing manifest model {self.preprocessing_manifest['model']!r} "
                f"does not match ONNX model {model_path.stem!r}"
            )
        configured_dataset = str(self.get_parameter("dataset").value)
        if (
            self.preprocessing_manifest is not None
            and self.preprocessing_manifest.get("dataset")
            not in (None, configured_dataset)
        ):
            raise ValueError(
                f"Preprocessing manifest dataset "
                f"{self.preprocessing_manifest['dataset']!r} does not match "
                f"configured dataset {configured_dataset!r}"
            )
        self.output_normalization = str(
            self.get_parameter("output_normalization").value
        )
        if self.output_normalization not in {"none", "l2"}:
            raise ValueError("output_normalization must be 'none' or 'l2'")
        self.synthetic_input = bool(self.get_parameter("synthetic_input").value)
        self.camera_rate_hz = float(self.get_parameter("camera_rate_hz").value)
        self.measurement_mode = str(
            self.get_parameter("measurement_mode").value
        ).lower()
        if self.measurement_mode not in {"frame", "saturated"}:
            raise ValueError("measurement_mode must be 'frame' or 'saturated'")
        if self.inference_trace_path and (
            self.measurement_mode != "frame" or not self.trace_output_frames
        ):
            raise ValueError("inference_trace_path requires frame mode and trace_output_frames")
        self.real_time_factor = float(
            self.get_parameter("real_time_factor").value
        )
        self.emulator = bool(self.get_parameter("emulator").value)
        self.scheduling_status = scheduling_status
        self.scheduling_error = scheduling_error
        if not math.isfinite(self.real_time_factor) or self.real_time_factor <= 0:
            raise ValueError("real_time_factor must be finite and positive")
        try:
            self.use_sim_time = bool(
                self.get_parameter("use_sim_time").value
            )
        except Exception:
            self.use_sim_time = False
        self.allow_spinning = bool(
            self.get_parameter("allow_spinning").value
        )
        self.realtime_priority = int(
            self.get_parameter("realtime_priority").value
        )
        self.lock_memory = bool(self.get_parameter("lock_memory").value)
        gpu_utilization_limit = float(
            self.get_parameter("gpu_utilization_limit").value
        )
        gpu_memory_limit_percent = float(
            self.get_parameter("gpu_memory_limit_percent").value
        )
        if (
            not math.isfinite(gpu_utilization_limit)
            or not 0 <= gpu_utilization_limit <= 100
            or not math.isfinite(gpu_memory_limit_percent)
            or not 0 <= gpu_memory_limit_percent <= 100
        ):
            raise ValueError("GPU utilization and memory limits must be percentages from 0 to 100")
        self.camera_data_timeout = float(
            self.get_parameter("camera_data_timeout").value
        )
        if not math.isfinite(self.camera_data_timeout) or self.camera_data_timeout <= 0:
            raise ValueError("camera_data_timeout must be finite and positive")
        if not math.isfinite(self.camera_rate_hz) or self.camera_rate_hz < 0:
            raise ValueError("camera_rate_hz must be finite and non-negative")
        if self.synthetic_input and self.camera_rate_hz == 0:
            raise ValueError("Synthetic input requires a positive camera_rate_hz")

        onnx_execution_mode = str(
            self.get_parameter("onnx_execution_mode").value
        ).lower()
        configured_providers = list(self.get_parameter("providers").value)
        available_providers = ort.get_available_providers()
        providers = resolve_onnx_providers(
            onnx_execution_mode,
            available_providers,
            configured_providers=configured_providers,
        )

        self.cpu_frequencies_before = sample_cpu_frequencies()
        self.temperatures_before = sample_temperatures()
        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        gpu_preflight_snapshot = sample_gpus()
        gpu_preflight = assess_gpu_load(
            gpu_preflight_snapshot,
            utilization_limit=gpu_utilization_limit,
            memory_limit_percent=gpu_memory_limit_percent,
        )
        session_options = ort.SessionOptions()
        self.enable_onnx_profiling = bool(
            self.get_parameter("enable_onnx_profiling").value
        )
        profile_prefix = None
        if self.enable_onnx_profiling:
            profile_dir = self.results_dir / model_path.stem
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_prefix = profile_dir / (
                "onnxruntime_profile_"
                + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            )
            session_options.enable_profiling = True
            session_options.profile_file_prefix = str(profile_prefix)
        if onnx_execution_mode == "cuda":
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        onnx_num_threads = int(self.get_parameter("onnx_num_threads").value)
        if onnx_num_threads > 0:
            session_options.intra_op_num_threads = onnx_num_threads
            session_options.inter_op_num_threads = 1
        allow_spinning = bool(self.get_parameter("allow_spinning").value)
        session_options.add_session_config_entry(
            "session.disable_spinning",
            "1" if not allow_spinning else "0",
        )
        self.cuda_stream = None
        self.cuda_io_binding = bool(self.get_parameter("cuda_io_binding").value)
        session_providers = providers
        if self.cuda_io_binding:
            if onnx_execution_mode != "cuda":
                raise ValueError("cuda_io_binding requires onnx_execution_mode=cuda")
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("cuda_io_binding requires CUDA-enabled PyTorch")
            self.cuda_device_id = torch.cuda.current_device()
            self.cuda_stream = torch.cuda.Stream(device=self.cuda_device_id)
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
            session_providers = [(
                "CUDAExecutionProvider",
                {
                    "device_id": self.cuda_device_id,
                    "user_compute_stream": str(self.cuda_stream.cuda_stream),
                },
            )]
        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=session_providers
        )
        active_providers = self.session.get_providers()
        expected_provider = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }.get(onnx_execution_mode)
        if expected_provider and expected_provider not in active_providers:
            raise RuntimeError(
                f"Strict ONNX {onnx_execution_mode.upper()} mode requested "
                f"{expected_provider}, but the session activated {active_providers}"
            )
        self.onnx_execution_mode = onnx_execution_mode
        self.onnx_requested_providers = providers
        self.gpu_preflight_snapshot = gpu_preflight_snapshot
        self.gpu_preflight = gpu_preflight
        self.onnxruntime_profile_prefix = profile_prefix
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active_providers:
            self.get_logger().warning(
                "CUDAExecutionProvider was requested but did not initialize; "
                f"active providers: {', '.join(active_providers)}"
            )
        self.model_input = self.session.get_inputs()[0]
        self.cuda_stream_latencies_ns = []
        self.cuda_input_upload_latencies_ns = []
        self.cuda_output_download_latencies_ns = []
        self.cuda_device_input = None
        self.cuda_input_signature = None
        if self.cuda_io_binding:
            self.session.disable_fallback()
            self.cuda_binding = self.session.io_binding()
            self.cuda_run_options = ort.RunOptions()
            self.cuda_run_options.add_run_config_entry(
                "disable_synchronize_execution_providers", "1"
            )
            model_outputs = self.session.get_outputs()
            self.cuda_output_names = [output.name for output in model_outputs]
            self.cuda_dynamic_outputs = any(
                output.shape is None
                or any(not isinstance(dimension, int) for dimension in output.shape)
                for output in model_outputs
            )
            for name in self.cuda_output_names:
                self.cuda_binding.bind_output(name, "cuda", self.cuda_device_id)
        input_type = str(self.model_input.type)
        numpy_dtype = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
        }.get(input_type, np.float32)
        self.saturated_model_tensor = np.zeros(
            tuple(
                dimension if isinstance(dimension, int) and dimension > 0 else 1
                for dimension in self.model_input.shape
            ),
            dtype=numpy_dtype,
        )
        self.model_path = model_path
        self.input_scale = float(self.get_parameter("input_scale").value)
        self.input_mean = list(
            self.get_parameter_or(
                "input_mean",
                Parameter("input_mean", Parameter.Type.DOUBLE_ARRAY, []),
            ).value
        )
        self.input_std = list(
            self.get_parameter_or(
                "input_std",
                Parameter("input_std", Parameter.Type.DOUBLE_ARRAY, []),
            ).value
        )
        if self.preprocessing_manifest is not None:
            preprocess_image(
                np.zeros((1, 1, 3), dtype=np.uint8),
                self.model_input.shape,
                operations=self.preprocessing_manifest["operations"],
            )
        self.warmup_runs = max(0, int(self.get_parameter("warmup_runs").value))
        self.warmup_convergence_window = max(
            0,
            int(self.get_parameter("warmup_convergence_window").value),
        )
        self.warmup_convergence_tolerance_percent = float(
            self.get_parameter("warmup_convergence_tolerance_percent").value
        )
        if (
            not math.isfinite(self.warmup_convergence_tolerance_percent)
            or self.warmup_convergence_tolerance_percent < 0
        ):
            raise ValueError("warmup_convergence_tolerance_percent must be non-negative")
        window = self.warmup_convergence_window
        default_budget = max(self.warmup_runs, 3 * window)
        self.warmup_max_runs = int(self.get_parameter("warmup_max_runs").value) or default_budget
        if window == 1 or self.warmup_max_runs < max(self.warmup_runs, 3 * window):
            raise ValueError("Warm-up window must be zero or at least two, with budget for three windows")
        self.warmup_converged = None if window == 0 else False
        self.camera_cpu_affinity = [
            int(cpu) for cpu in str(self.get_parameter("camera_cpu_affinity_cpus").value).split(",")
            if cpu
        ]
        self.min_measurement_seconds = float(self.get_parameter("min_measurement_seconds").value)
        if not math.isfinite(self.min_measurement_seconds) or self.min_measurement_seconds < 0:
            raise ValueError("min_measurement_seconds must be finite and non-negative")
        self.measurement_started_ns = None
        self.measurement_elapsed_ns = 0
        self.iterations = int(self.get_parameter("iterations").value)
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

        self.bridge = CvBridge()
        self.completed_warmups = 0
        self.warmup_latencies_ns = []
        self.startup_drop_frames = max(
            0, int(self.get_parameter("startup_drop_frames").value)
        )
        self.startup_frames_dropped = 0
        self.ready = False
        self.latencies_ns = []
        self.onnxruntime_latencies_ns = []
        self.end_to_end_latencies_ns = []
        self.callback_latencies_ns = []
        self.source_to_callback_latencies_ns = []
        self.end_to_end_timestamp_frames = 0
        self.clock_inconsistency_frames = 0
        self.clock_error_message = None
        self.frame_observations = []
        self.arrival_timestamps_ns = []
        self.cpu_telemetry_monitor = CpuTelemetryMonitor(include_machine_telemetry=True)
        self.failed_frames = 0
        self.last_frame_error = None
        self.record_saved = False
        self.shutdown_timer = None
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.last_camera_data_at = time.monotonic()
        self.camera_data_timeout_expired = False
        self.output_publisher = self.create_publisher(
            Float32MultiArray, str(self.get_parameter("output_topic").value), 10
        )
        self.image_subscription_queue_depth = 5 if self.synthetic_input else 1
        # Reliable delivery keeps the local synthetic stream's large image
        # fragments intact; physical sensor streams retain low-latency best effort.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.image_subscription_queue_depth,
            reliability=(
                ReliabilityPolicy.RELIABLE
                if self.synthetic_input
                else ReliabilityPolicy.BEST_EFFORT
            ),
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.on_image,
            image_qos,
        )
        if self.measurement_mode == "frame":
            self.camera_data_timeout_timer = self.create_timer(
                min(1.0, self.camera_data_timeout),
                self.check_camera_data_timeout,
                clock=self.steady_clock,
            )
        # Transient-local readiness lets the launcher start acquisition only after
        # the ONNX session and image subscription are ready, even if it subscribes late.
        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_publisher = self.create_publisher(
            Bool, str(self.get_parameter("ready_topic").value), ready_qos
        )
        if self.inference_trace_path:
            self.inference_trace = open(self.inference_trace_path, "x", encoding="utf-8")
        ready_message = Bool()
        ready_message.data = True
        self.ready_publisher.publish(ready_message)
        self.ready = True
        self.ready_timer = self.create_timer(0.1, self.publish_readiness, clock=self.steady_clock)
        self.saturated_timer = None
        if self.measurement_mode == "saturated":
            self.saturated_timer = self.create_timer(
                0.001, self.run_saturated_benchmark, clock=self.steady_clock
            )
        self.get_logger().info(
            f"Loaded {model_path.name}; providers: {', '.join(active_providers)}"
        )
        self.get_logger().info(
            f"[NN_BOT_READY] model/session initialized; subscribed to "
            f"{self.input_topic}"
        )

    def destroy_node(self):
        try:
            return super().destroy_node()
        finally:
            if self.inference_trace is not None:
                self.inference_trace.close()

    def publish_readiness(self):
        if not self.ready:
            return
        ready_message = Bool()
        ready_message.data = True
        self.ready_publisher.publish(ready_message)
        if len(self.latencies_ns) > 0 or self.startup_frames_dropped > 0:
            if self.ready_timer is not None:
                self.ready_timer.cancel()
                self.ready_timer = None

    def warming_up(self):
        return self.completed_warmups < self.warmup_runs or (
            self.warmup_converged is False and self.completed_warmups < self.warmup_max_runs
        )

    def record_warmup(self, elapsed_ns):
        self.completed_warmups += 1
        self.warmup_latencies_ns.append(elapsed_ns)
        window = self.warmup_convergence_window
        if not window or self.completed_warmups < max(self.warmup_runs, 3 * window):
            return
        tail = self.warmup_latencies_ns[-3 * window:]
        medians = [float(np.median(tail[i:i + window])) for i in range(0, len(tail), window)]
        tolerance = self.warmup_convergence_tolerance_percent / 100.0
        self.warmup_converged = all(
            abs(current - previous) <= previous * tolerance
            for previous, current in zip(medians, medians[1:])
        )

    def start_measurement(self):
        self.cpu_telemetry_monitor.start()
        self.measurement_started_ns = time.perf_counter_ns()

    def measurement_complete(self):
        self.measurement_elapsed_ns = time.perf_counter_ns() - self.measurement_started_ns
        return (
            len(self.latencies_ns) >= self.iterations
            and self.measurement_elapsed_ns >= self.min_measurement_seconds * 1_000_000_000
        )

    def run_saturated_benchmark(self):
        while not self.record_saved:
            warming_up = self.warming_up()
            if not warming_up and not self.cpu_telemetry_monitor.started:
                self.start_measurement()
            outputs, elapsed_ns, cuda_timings = self.run_inference(
                self.saturated_model_tensor
            )
            if warming_up:
                self.record_warmup(elapsed_ns)
                continue

            self.latencies_ns.append(elapsed_ns)
            self.onnxruntime_latencies_ns.append(elapsed_ns)
            if cuda_timings is not None:
                self.record_cuda_timings(cuda_timings)
            process_model_output(outputs[0], self.output_normalization)
            if self.measurement_complete():
                if self.saturated_timer is not None:
                    self.saturated_timer.cancel()
                    self.saturated_timer = None
                self.finish_benchmark(self.saturated_model_tensor.shape, outputs[0].shape)

    def run_inference(self, model_tensor):
        started_ns = time.perf_counter_ns()
        if self.cuda_stream is None:
            outputs = self.session.run(None, {self.model_input.name: model_tensor})
            return outputs, time.perf_counter_ns() - started_ns, None

        host_input = np.ascontiguousarray(model_tensor)
        signature = (host_input.shape, host_input.dtype)
        if signature == self.cuda_input_signature:
            self.cuda_device_input.update_inplace(host_input)
        else:
            device_input = ort.OrtValue.ortvalue_from_numpy(
                host_input, "cuda", self.cuda_device_id
            )
            self.cuda_binding.bind_ortvalue_input(self.model_input.name, device_input)
            self.cuda_device_input = device_input
            self.cuda_input_signature = signature
        if self.cuda_dynamic_outputs:
            # ORT retains the previous output allocation in a reused binding.
            self.cuda_binding.clear_binding_outputs()
            for name in self.cuda_output_names:
                self.cuda_binding.bind_output(name, "cuda", self.cuda_device_id)
        self.cuda_binding.synchronize_inputs()
        uploaded_ns = time.perf_counter_ns()
        self.cuda_start.record(self.cuda_stream)
        try:
            self.session.run_with_iobinding(self.cuda_binding, self.cuda_run_options)
        finally:
            self.cuda_end.record(self.cuda_stream)
            self.cuda_end.synchronize()
        cuda_elapsed_ns = round(self.cuda_start.elapsed_time(self.cuda_end) * 1_000_000)
        download_started_ns = time.perf_counter_ns()
        outputs = self.cuda_binding.copy_outputs_to_cpu()
        finished_ns = time.perf_counter_ns()
        return outputs, finished_ns - started_ns, (
            uploaded_ns - started_ns, cuda_elapsed_ns, finished_ns - download_started_ns
        )

    def record_cuda_timings(self, timings):
        upload_ns, stream_ns, download_ns = timings
        self.cuda_input_upload_latencies_ns.append(upload_ns)
        self.cuda_stream_latencies_ns.append(stream_ns)
        self.cuda_output_download_latencies_ns.append(download_ns)

    def on_image(self, message):
        # Discard initial frames after model readiness, before any timing or
        # benchmark counters are updated. QoS depth bounds any queued backlog.
        if not self.ready or self.measurement_mode == "saturated":
            return
        key = frame_key(message) if self.trace_output_frames else None
        if self.inference_trace is not None:
            write_observation(self.inference_trace, key, stage="arrival")
        self.last_camera_data_at = time.monotonic()
        if self.startup_frames_dropped < self.startup_drop_frames:
            self.startup_frames_dropped += 1
            return

        warming_up = self.warming_up()
        measuring = not warming_up and not self.record_saved
        if measuring and self.measurement_started_ns is None:
            self.start_measurement()

        callback_started_ns = time.perf_counter_ns()
        arrival_ros_ns = self.get_clock().now().nanoseconds
        source_stamp_ns = key[1] if key is not None else message_stamp_ns(message)
        observation = None
        if measuring:
            self.arrival_timestamps_ns.append(callback_started_ns)
            observation = {
                "source_ros_ns": source_stamp_ns or None,
                "arrival_ros_ns": arrival_ros_ns,
                "arrival_monotonic_ns": callback_started_ns,
                "publish_ros_ns": None,
                "publish_monotonic_ns": None,
            }
            self.frame_observations.append(observation)
        published = False
        try:
            started = time.perf_counter_ns()
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
            model_tensor = preprocess_image(
                image,
                self.model_input.shape,
                self.input_scale,
                self.input_mean,
                self.input_std,
                operations=(
                    self.preprocessing_manifest["operations"]
                    if self.preprocessing_manifest is not None
                    else None
                ),
            )
            outputs, inference_elapsed_ns, cuda_timings = self.run_inference(model_tensor)
            output = process_model_output(outputs[0], self.output_normalization)

            message_out = Float32MultiArray()
            message_out.layout.dim = [
                MultiArrayDimension(
                    label=f"dim_{index}",
                    size=int(size),
                    stride=math.prod(output.shape[index + 1 :]),
                )
                for index, size in enumerate(output.shape)
            ]
            message_out.data = output.reshape(-1).tolist()
            if key is not None:
                tag_output(message_out, key)
            self.output_publisher.publish(message_out)
            published = True
            if self.record_saved:
                return
            elapsed_ns = time.perf_counter_ns() - started
            published_ros_ns = self.get_clock().now().nanoseconds
            callback_elapsed_ns = time.perf_counter_ns() - callback_started_ns
            if observation is not None:
                observation["publish_ros_ns"] = published_ros_ns
                observation["publish_monotonic_ns"] = callback_started_ns + callback_elapsed_ns
            try:
                header_latency_ns = header_to_publish_latency_ns(
                    source_stamp_ns, arrival_ros_ns, published_ros_ns
                )
            except ValueError as error:
                self.clock_inconsistency_frames += 1
                self.clock_error_message = str(error)
                self.get_logger().error(self.clock_error_message)
                header_latency_ns = None

            if warming_up:
                self.record_warmup(inference_elapsed_ns)
                return

            self.latencies_ns.append(elapsed_ns)
            self.onnxruntime_latencies_ns.append(inference_elapsed_ns)
            if cuda_timings is not None:
                self.record_cuda_timings(cuda_timings)
            self.callback_latencies_ns.append(callback_elapsed_ns)
            if header_latency_ns is not None:
                self.end_to_end_latencies_ns.append(header_latency_ns)
                self.source_to_callback_latencies_ns.append(arrival_ros_ns - source_stamp_ns)
                self.end_to_end_timestamp_frames += 1
            if self.measurement_complete():
                self.finish_benchmark(model_tensor.shape, output.shape)
        except Exception as error:
            self.failed_frames += 1
            self.last_frame_error = f"{type(error).__name__}: {error}"
            self.get_logger().error(f"Inference failed: {error}")
        finally:
            if published and self.inference_trace is not None:
                write_observation(self.inference_trace, key, stage="publish")

    def check_camera_data_timeout(self):
        if time.monotonic() - self.last_camera_data_at < self.camera_data_timeout:
            return
        self.camera_data_timeout_expired = True
        self.get_logger().warning(
            "No images received on "
            f"{self.input_topic}. Start a camera publisher or use --emulate-camera."
        )
        rclpy.try_shutdown()

    def save_record(self, input_shape, output_shape):
        device_type = device_name()
        if input_shape[1] in (1, 3, 4):
            input_batch, input_channels, input_height, input_width = input_shape
        else:
            input_batch, input_height, input_width, input_channels = input_shape
        deadline_ms = float(self.get_parameter("deadline_ms").value)
        deadline_ns = int(round(deadline_ms * 1_000_000))
        deadline_missed_frames = int(sum(1 for lat in self.latencies_ns if lat > deadline_ns))
        cpu_telemetry = self.cpu_telemetry_monitor.stop_and_report()
        cpu_frequencies = sample_cpu_frequencies()
        temperatures = sample_temperatures()
        cpu_frequencies_before = self.cpu_frequencies_before
        temperatures_before = self.temperatures_before
        condition_warnings = list(dict.fromkeys(
            measurement_condition_warnings(cpu_frequencies_before, temperatures_before)
            + measurement_condition_warnings(cpu_frequencies, temperatures)
        ))
        if self.scheduling_status in {"requested_but_unavailable", "partial"}:
            condition_warnings.append(
                f"Realtime scheduling/locking not fully available: {self.scheduling_error}"
            )
        observed_input_rate_hz = (
            1_000_000_000 / float(np.mean(np.diff(self.arrival_timestamps_ns)))
            if len(self.arrival_timestamps_ns) > 1
            else None
        )
        stamps = [frame["source_ros_ns"] for frame in self.frame_observations]
        usable_stamps = len(stamps) > 1 and all(stamp and stamp > 0 for stamp in stamps)
        if usable_stamps:
            usable_stamps = all(previous < current for previous, current in zip(stamps, stamps[1:]))
        observed_ros_clock_factor = None
        if self.use_sim_time and usable_stamps:
            ros_span = self.frame_observations[-1]["arrival_ros_ns"] - self.frame_observations[0]["arrival_ros_ns"]
            monotonic_span = self.arrival_timestamps_ns[-1] - self.arrival_timestamps_ns[0]
            if ros_span > 0 and monotonic_span > 0:
                observed_ros_clock_factor = ros_span / monotonic_span
        check_camera_rate = self.measurement_mode == "frame" and self.camera_rate_hz > 0
        dropped_frames = None
        if check_camera_rate and usable_stamps and not self.clock_inconsistency_frames:
            dropped_frames = sum(
                dropped_frames_between(previous, current, self.camera_rate_hz)
                for previous, current in zip(stamps, stamps[1:])
            )
        dropped_frame_fraction_value = None
        dropped_frame_tolerance_met = None
        if dropped_frames is not None:
            dropped_frame_fraction_value = dropped_frame_fraction(dropped_frames, len(stamps))
            dropped_frame_tolerance_met = dropped_frames_within_tolerance(dropped_frames, len(stamps))
        measurement_quality_errors = []
        if self.failed_frames:
            measurement_quality_errors.append(
                f"{self.failed_frames} image frame(s) failed: {self.last_frame_error}"
            )
        if self.clock_inconsistency_frames:
            measurement_quality_errors.append(
                f"{self.clock_inconsistency_frames} image frame(s) had inconsistent ROS clocks: "
                f"{self.clock_error_message}"
            )
        if len(self.latencies_ns) < self.iterations:
            measurement_quality_errors.append(
                f"collected {len(self.latencies_ns)} of {self.iterations} requested inference samples"
            )
        if dropped_frames and not dropped_frame_tolerance_met:
            measurement_quality_errors.append(
                f"{dropped_frames} estimated camera frame(s) were dropped "
                f"({dropped_frame_fraction_value:.3%}; tolerance "
                f"{DROPPED_FRAME_TOLERANCE_FRACTION:.3%})"
            )
        input_rate_met = None
        if check_camera_rate:
            if dropped_frames is None:
                measurement_quality_errors.append(
                    "Camera drop estimates require increasing, clock-consistent source timestamps"
                )
            clock_factor = observed_ros_clock_factor if self.use_sim_time else 1.0
            if clock_factor is None or observed_input_rate_hz is None:
                measurement_quality_errors.append("Camera wall-rate target could not be evaluated")
            else:
                expected_wall_rate_hz = self.camera_rate_hz * clock_factor
                input_rate_met = observed_input_rate_hz >= expected_wall_rate_hz * 0.95
                if not input_rate_met:
                    measurement_quality_errors.append(
                        f"Observed input rate {observed_input_rate_hz:.2f} Hz was below "
                        f"95% of the expected wall rate {expected_wall_rate_hz:g} Hz"
                    )
        consecutive_late_publishes = 0
        max_consecutive_late_publishes = 0
        for frame in self.frame_observations:
            source_ns = frame["source_ros_ns"]
            publish_ns = frame["publish_ros_ns"]
            if source_ns and publish_ns and source_ns <= frame["arrival_ros_ns"] <= publish_ns:
                frame["source_to_publish_deadline_missed"] = publish_ns - source_ns > deadline_ns
            else:
                frame["source_to_publish_deadline_missed"] = None
            if frame["source_to_publish_deadline_missed"]:
                consecutive_late_publishes += 1
                max_consecutive_late_publishes = max(
                    max_consecutive_late_publishes, consecutive_late_publishes
                )
            else:
                consecutive_late_publishes = 0
        onnxruntime_timing = timing_stats(self.onnxruntime_latencies_ns)
        onnxruntime_deadline_misses = int(
            sum(1 for lat in self.onnxruntime_latencies_ns if lat > deadline_ns)
        )
        end_to_end_timing = (
            timing_stats(self.end_to_end_latencies_ns)
            if self.end_to_end_latencies_ns
            else None
        )
        if onnxruntime_timing["p99"] is not None and onnxruntime_timing["p99"] > deadline_ns:
            measurement_quality_errors.append(
                f"ONNX Runtime p99 {onnxruntime_timing['p99'] / 1_000_000:.3f} ms "
                f"exceeds deadline {deadline_ms:.3f} ms "
                f"({onnxruntime_deadline_misses} samples exceeded)"
            )
        if (
            end_to_end_timing is not None
            and end_to_end_timing["p99"] is not None
            and end_to_end_timing["p99"] > deadline_ns
        ):
            measurement_quality_errors.append(
                f"end-to-end p99 {end_to_end_timing['p99'] / 1_000_000:.3f} ms "
                f"exceeds deadline {deadline_ms:.3f} ms"
            )
        if self.gpu_preflight["busy"]:
            measurement_quality_errors.append("GPU preflight detected a busy GPU")
        measurement_quality_errors.extend(condition_warnings)
        warmup_median_ns = (
            float(np.median(self.warmup_latencies_ns)) if self.warmup_latencies_ns else None
        )
        if self.warmup_converged is False:
            measurement_quality_errors.append(
                f"Warm-up did not converge within {self.warmup_max_runs} runs"
            )
        measurement_conditions_valid = not measurement_quality_errors
        preprocessing_provenance_valid = self.preprocessing_manifest is not None
        performance_quality_errors = list(measurement_quality_errors)
        if not preprocessing_provenance_valid:
            performance_quality_errors.append(
                "No preprocessing manifest; legacy parameter preprocessing is not verified"
            )
        if self.enable_onnx_profiling:
            performance_quality_errors.append(
                "ONNX Runtime profiling is enabled; latency samples are diagnostic only"
            )
        performance_quality_errors = list(dict.fromkeys(performance_quality_errors))
        performance_measurement_valid = not performance_quality_errors
        active_providers = self.session.get_providers()
        onnxruntime_profile = None
        if self.enable_onnx_profiling:
            profile_path = self.session.end_profiling()
            onnxruntime_profile = summarize_onnxruntime_profile(profile_path)
        record = {
            "model_name": self.model_path.stem,
            "dataset": str(self.get_parameter("dataset").value),
            "device_type": device_type,
            "board": device_type,
            "architecture": platform.machine(),
            "os_version": platform.platform(),
            "valid": False,
            "profile_version": 8,
            "run_complete": (
                len(self.latencies_ns) == self.iterations and self.failed_frames == 0
            ),
            "measurement_conditions_valid": measurement_conditions_valid,
            "preprocessing_provenance_valid": preprocessing_provenance_valid,
            "preprocessing_provenance_status": (
                "model_manifest" if preprocessing_provenance_valid
                else "legacy_ros_parameters_unverified"
            ),
            "inference_correctness_valid": None,
            "inference_validation_status": "pending_pytorch_onnx_comparison",
            "performance_measurement_valid": performance_measurement_valid,
            "measurement_quality_errors": measurement_quality_errors,
            "performance_quality_errors": performance_quality_errors,
            "emulator": self.emulator,
            "emulator_requested": bool(
                self.get_parameter("emulator").value
            ),
            "use_sim_time": self.use_sim_time,
            "real_time_factor": self.real_time_factor,
            "measurement_mode": self.measurement_mode,
            "allow_spinning": self.allow_spinning,
            "scheduling": {
                "status": self.scheduling_status,
                "realtime_priority": self.realtime_priority,
                "lock_memory": self.lock_memory,
                "error": self.scheduling_error,
            },
            "warmup_convergence": {
                "window": self.warmup_convergence_window,
                "tolerance_percent": self.warmup_convergence_tolerance_percent,
                "median_ns": warmup_median_ns,
                "converged": self.warmup_converged,
                "maximum_runs": self.warmup_max_runs,
                "stable_window_transitions_required": 2,
            },
            "synthetic_input": self.synthetic_input,
            "input_source": "synthetic_camera" if self.synthetic_input else "sensor_topic",
            "camera_rate_hz": self.camera_rate_hz if check_camera_rate else None,
            "observed_ros_clock_factor": observed_ros_clock_factor,
            "frame_observations": self.frame_observations,
            "dropped_frames_scope": (
                "estimated_from_source_stamp_gaps" if dropped_frames is not None else None
            ),
            "observed_input_rate_hz": observed_input_rate_hz,
            "input_rate_target_met": input_rate_met,
            "input_rate_tolerance_fraction": 0.95,
            "dropped_frame_fraction": dropped_frame_fraction_value,
            "dropped_frame_tolerance_fraction": (
                DROPPED_FRAME_TOLERANCE_FRACTION if check_camera_rate else None
            ),
            "dropped_frame_tolerance_met": dropped_frame_tolerance_met,
            "iterations": self.iterations,
            "measured_iterations": len(self.latencies_ns),
            "min_measurement_seconds": self.min_measurement_seconds,
            "measurement_elapsed_seconds": self.measurement_elapsed_ns / 1_000_000_000,
            "startup_drop_frames": self.startup_drop_frames,
            "startup_frames_dropped": self.startup_frames_dropped,
            "readiness_handshake": True,
            "image_subscription_queue_depth": self.image_subscription_queue_depth,
            "failed_frames": self.failed_frames,
            "clock_inconsistency_frames": self.clock_inconsistency_frames,
            "clock_error_message": self.clock_error_message,
            "dropped_frames": dropped_frames,
            "arrival_frame_count": len(self.arrival_timestamps_ns),
            "last_frame_error": self.last_frame_error,
            "error_message": "; ".join(performance_quality_errors)
            or "Inference correctness validation has not run",
            "duration": int(round(float(np.mean(self.latencies_ns)))),
            "deadline_ms": deadline_ms,
            "deadline_missed_frames": deadline_missed_frames,
            "pipeline_deadline_missed_frames": (
                deadline_missed_frames if self.measurement_mode == "frame" else None
            ),
            "max_consecutive_late_publishes": (
                max_consecutive_late_publishes if self.end_to_end_latencies_ns else None
            ),
            "source_to_publish_deadline_missed_frames": (
                sum(latency > deadline_ns for latency in self.end_to_end_latencies_ns)
                if self.end_to_end_latencies_ns else None
            ),
            "missed_deadlines": deadline_missed_frames,
            "cpu_runs": None,
            "gpu_runs": None,
            "npu_duration": None,
            "npu_min_duration": None,
            "npu_max_duration": None,
            "npu_std_dev": None,
            "npu_runs": 0,
            "npu_error": None,
            "unit": device_type,
            "runtime": "ONNX Runtime",
            "onnxruntime_provider": (
                self.onnx_requested_providers[0]
                if self.onnx_execution_mode in {"cpu", "cuda"}
                or len(active_providers) == 1
                else None
            ),
            "onnxruntime_execution_mode": self.onnx_execution_mode,
            "onnxruntime_requested_providers": self.onnx_requested_providers,
            "onnxruntime_active_providers": active_providers,
            "onnxruntime_provider_status": (
                (
                    "strict CUDA execution; CPU EP fallback disabled"
                    if self.onnx_execution_mode == "cuda"
                    else "strict CPU-only session"
                )
                if self.onnx_execution_mode in {"cpu", "cuda"}
                else "active session providers; per-node execution provider not measured"
            ),
            "onnxruntime_profiling_enabled": self.enable_onnx_profiling,
            "onnxruntime_profile": onnxruntime_profile,
            "model_format": "onnx",
            "output_normalization": self.output_normalization,
            "preprocessing_manifest": (
                str(self.preprocessing_manifest_path)
                if self.preprocessing_manifest is not None
                else None
            ),
            "preprocessing": (
                self.preprocessing_manifest
                if self.preprocessing_manifest is not None
                else {
                    "source": "ros_parameters",
                    "resize": "legacy_nearest",
                    "color_order": "RGB",
                    "scale": self.input_scale,
                    "mean": self.input_mean,
                    "std": self.input_std,
                }
            ),
            "input_topic": str(self.get_parameter("input_topic").value),
            "trace_output_frames": self.trace_output_frames,
            "inference_trace_path": self.inference_trace_path or None,
            "inference_time_scope": "image_preprocessing_through_output_publish",
            "onnxruntime_inference_time_scope": (
                "host_io_binding_upload_inference_download"
                if self.cuda_io_binding else "session_run_only"
            ),
            "model_size_bytes": self.model_path.stat().st_size,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "latency_ns": timing_stats(self.latencies_ns, deadline_ns=deadline_ns),
            "onnxruntime_latency_ns": timing_stats(self.onnxruntime_latencies_ns),
            "cuda_io_binding": self.cuda_io_binding,
            "onnxruntime_cuda_stream_latency_ns": (
                timing_stats(self.cuda_stream_latencies_ns) if self.cuda_io_binding else None
            ),
            "onnxruntime_cuda_stream_latency_scope": (
                "cuda_compute_stream_with_device_io" if self.cuda_io_binding else None
            ),
            "onnxruntime_cuda_input_upload_ns": (
                timing_stats(self.cuda_input_upload_latencies_ns) if self.cuda_io_binding else None
            ),
            "onnxruntime_cuda_output_download_ns": (
                timing_stats(self.cuda_output_download_latencies_ns) if self.cuda_io_binding else None
            ),
            "end_to_end_latency_ns": end_to_end_timing,
            "end_to_end_latency_scope": "image_header_stamp_to_output_publish",
            "end_to_end_header_timestamp_frames": self.end_to_end_timestamp_frames,
            "callback_latency_ns": (
                timing_stats(self.callback_latencies_ns) if self.callback_latencies_ns else None
            ),
            "callback_latency_scope": "callback_entry_to_output_publish",
            "source_to_callback_latency_ns": (
                timing_stats(self.source_to_callback_latencies_ns)
                if self.source_to_callback_latencies_ns
                else None
            ),
            "source_to_callback_latency_scope": "image_header_stamp_to_callback_entry",
            "measurement_condition_warnings": condition_warnings,
            "gpu_preflight": {
                "snapshot": self.gpu_preflight_snapshot,
                "assessment": self.gpu_preflight,
            },
            "in_dim_0": input_batch,
            "in_dim_1": input_height,
            "in_dim_2": input_width,
            "in_dim_3": input_channels,
            "device_analytics": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "python_version": platform.python_version(),
                "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
                "cpu_cores": os.cpu_count(),
                "cpu_affinity": get_cpu_affinity(),
                "camera_cpu_affinity": self.camera_cpu_affinity or None,
                "cpu_frequencies": cpu_frequencies,
                "cpu_frequencies_before": cpu_frequencies_before,
                "temperatures": temperatures,
                "temperatures_before": temperatures_before,
                "cpu_telemetry": cpu_telemetry,
                "onnxruntime_version": ort.__version__,
                "onnx_num_threads": int(self.get_parameter("onnx_num_threads").value) or None,
                "providers": self.session.get_providers(),
            },
        }
        record["warmup_runs"] = self.warmup_runs
        record["completed_warmups"] = self.completed_warmups
        output_dir = self.results_dir / self.model_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = str(self.get_parameter("dataset").value)
        output_path = output_dir / f"robot_{dataset}_{socket.gethostname()}.json"
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        self.get_logger().info(f"Saved {len(self.latencies_ns)}-sample benchmark to {output_path}")

    def finish_benchmark(self, input_shape, output_shape):
        self.save_record(list(input_shape), list(output_shape))
        self.record_saved = True
        if self.get_parameter("exit_after_benchmark").value:
            self.shutdown_timer = self.create_timer(
                0.1, self.stop_after_benchmark, clock=self.steady_clock
            )

    def stop_after_benchmark(self):
        if self.ready_timer is not None:
            self.ready_timer.cancel()
            self.ready_timer = None
        if self.shutdown_timer is not None:
            self.shutdown_timer.cancel()
            self.shutdown_timer = None
        rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = InferenceNode()
        executor.add_node(node)
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        if node.camera_data_timeout_expired:
            return CAMERA_DATA_TIMEOUT_EXIT_CODE
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
