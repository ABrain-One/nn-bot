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
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension

from ab.bot.runtime import (
    CAMERA_DATA_TIMEOUT_EXIT_CODE,
    device_name,
    dropped_frames_between,
    get_cpu_affinity,
    header_to_publish_latency_ns,
    load_preprocessing_manifest,
    message_stamp_ns,
    measurement_condition_warnings,
    preprocess_image,
    process_model_output,
    sample_cpu_frequencies,
    sample_temperatures,
    set_cpu_affinity,
    timing_stats,
)


class InferenceNode(Node):
    def __init__(self):
        super().__init__("nn_bot_inference")
        self.declare_parameter("model_path", "")
        self.declare_parameter("input_topic", "/camera/image_raw")
        self.declare_parameter("camera_data_timeout", 90.0)
        self.declare_parameter("output_topic", "/nn_bot/output")
        self.declare_parameter("preprocessing_manifest", "")
        self.declare_parameter("output_normalization", "none")
        self.declare_parameter("results_dir", "stats")
        self.declare_parameter("dataset", "custom")
        self.declare_parameter("synthetic_input", False)
        self.declare_parameter("camera_rate_hz", 0.0)
        self.declare_parameter("warmup_runs", 10)
        self.declare_parameter("iterations", 500)
        self.declare_parameter("deadline_ms", 33.0)
        self.declare_parameter("cpu_affinity", -1)
        self.declare_parameter("onnx_num_threads", 0)
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
        if (
            self.preprocessing_manifest is not None
            and self.preprocessing_manifest.get("model") not in (None, model_path.stem)
        ):
            raise ValueError(
                f"Preprocessing manifest model {self.preprocessing_manifest['model']!r} "
                f"does not match ONNX model {model_path.stem!r}"
            )
        self.output_normalization = str(
            self.get_parameter("output_normalization").value
        )
        if self.output_normalization not in {"none", "l2"}:
            raise ValueError("output_normalization must be 'none' or 'l2'")
        self.synthetic_input = bool(self.get_parameter("synthetic_input").value)
        self.camera_rate_hz = float(self.get_parameter("camera_rate_hz").value)
        self.camera_data_timeout = float(
            self.get_parameter("camera_data_timeout").value
        )
        if not math.isfinite(self.camera_data_timeout) or self.camera_data_timeout <= 0:
            raise ValueError("camera_data_timeout must be finite and positive")
        if self.synthetic_input and (
            not math.isfinite(self.camera_rate_hz) or self.camera_rate_hz <= 0
        ):
            raise ValueError("camera_rate_hz must be finite and positive for synthetic input")

        requested_providers = list(self.get_parameter("providers").value)
        available_providers = ort.get_available_providers()
        providers = [name for name in requested_providers if name in available_providers]
        if not providers:
            raise RuntimeError(
                f"None of the requested ONNX Runtime providers are available: "
                f"{requested_providers}; available: {available_providers}"
            )

        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        session_options = ort.SessionOptions()
        onnx_num_threads = int(self.get_parameter("onnx_num_threads").value)
        if onnx_num_threads > 0:
            session_options.intra_op_num_threads = onnx_num_threads
            session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        active_providers = self.session.get_providers()
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in active_providers:
            self.get_logger().warning(
                "CUDAExecutionProvider was requested but did not initialize; "
                f"active providers: {', '.join(active_providers)}"
            )
        self.model_input = self.session.get_inputs()[0]
        self.model_path = model_path
        self.input_scale = float(self.get_parameter("input_scale").value)
        self.input_mean = list(self.get_parameter("input_mean").value)
        self.input_std = list(self.get_parameter("input_std").value)
        if self.preprocessing_manifest is not None:
            preprocess_image(
                np.zeros((1, 1, 3), dtype=np.uint8),
                self.model_input.shape,
                operations=self.preprocessing_manifest["operations"],
            )
        self.warmup_runs = max(0, int(self.get_parameter("warmup_runs").value))
        self.iterations = int(self.get_parameter("iterations").value)
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")

        self.results_dir = Path(self.get_parameter("results_dir").value).expanduser()
        self.bridge = CvBridge()
        self.completed_warmups = 0
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
        self.previous_synthetic_stamp_ns = None
        self.dropped_frames = 0
        self.arrival_timestamps_ns = []
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
        # Keep only the newest pending camera frame. This limits stale backlog when
        # an external camera is already publishing while the model is initialized.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.on_image,
            image_qos,
        )
        self.camera_data_timeout_timer = self.create_timer(
            min(1.0, self.camera_data_timeout), self.check_camera_data_timeout
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
        ready_message = Bool()
        ready_message.data = True
        self.ready_publisher.publish(ready_message)
        self.ready = True
        self.ready_timer = self.create_timer(0.1, self.publish_readiness)
        self.get_logger().info(
            f"Loaded {model_path.name}; providers: {', '.join(active_providers)}"
        )
        self.get_logger().info(
            f"[NN_BOT_READY] model/session initialized; subscribed to "
            f"{self.input_topic}"
        )

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

    def on_image(self, message):
        # Discard initial frames after model readiness, before any timing or
        # benchmark counters are updated. QoS depth=1 bounds any queued backlog.
        if not self.ready:
            return
        self.last_camera_data_at = time.monotonic()
        if self.startup_frames_dropped < self.startup_drop_frames:
            self.startup_frames_dropped += 1
            return

        callback_started_ns = time.perf_counter_ns()
        arrival_ros_ns = self.get_clock().now().nanoseconds
        source_stamp_ns = message_stamp_ns(message)
        self.arrival_timestamps_ns.append(callback_started_ns)
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
            inference_started_ns = time.perf_counter_ns()
            outputs = self.session.run(None, {self.model_input.name: model_tensor})
            inference_elapsed_ns = time.perf_counter_ns() - inference_started_ns
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
            self.output_publisher.publish(message_out)
            elapsed_ns = time.perf_counter_ns() - started
            published_ros_ns = self.get_clock().now().nanoseconds
            callback_elapsed_ns = time.perf_counter_ns() - callback_started_ns
            try:
                header_latency_ns = header_to_publish_latency_ns(
                    source_stamp_ns, arrival_ros_ns, published_ros_ns
                )
                if self.synthetic_input and source_stamp_ns > 0:
                    if self.previous_synthetic_stamp_ns is not None:
                        self.dropped_frames += dropped_frames_between(
                            self.previous_synthetic_stamp_ns,
                            source_stamp_ns,
                            self.camera_rate_hz,
                        )
                    self.previous_synthetic_stamp_ns = source_stamp_ns
            except ValueError as error:
                self.clock_inconsistency_frames += 1
                self.clock_error_message = str(error)
                self.get_logger().error(self.clock_error_message)
                header_latency_ns = None

            if self.completed_warmups < self.warmup_runs:
                self.completed_warmups += 1
                return

            if len(self.latencies_ns) < self.iterations:
                self.latencies_ns.append(elapsed_ns)
                self.onnxruntime_latencies_ns.append(inference_elapsed_ns)
                self.callback_latencies_ns.append(callback_elapsed_ns)
                if header_latency_ns is not None:
                    self.end_to_end_latencies_ns.append(header_latency_ns)
                    self.source_to_callback_latencies_ns.append(
                        arrival_ros_ns - source_stamp_ns
                    )
                    self.end_to_end_timestamp_frames += 1
                if len(self.latencies_ns) == self.iterations and not self.record_saved:
                    self.save_record(list(model_tensor.shape), list(output.shape))
                    self.record_saved = True
                    if self.get_parameter("exit_after_benchmark").value:
                        self.shutdown_timer = self.create_timer(0.1, self.stop_after_benchmark)
        except Exception as error:
            self.failed_frames += 1
            self.last_frame_error = f"{type(error).__name__}: {error}"
            self.get_logger().error(f"Inference failed: {error}")

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
        cpu_frequencies = sample_cpu_frequencies()
        temperatures = sample_temperatures()
        condition_warnings = measurement_condition_warnings(
            cpu_frequencies, temperatures
        )
        dropped_frames = self.dropped_frames if self.synthetic_input else None
        validity_errors = []
        if self.failed_frames:
            validity_errors.append(
                f"{self.failed_frames} image frame(s) failed: {self.last_frame_error}"
            )
        if self.clock_inconsistency_frames:
            validity_errors.append(
                f"{self.clock_inconsistency_frames} image frame(s) had inconsistent ROS clocks: "
                f"{self.clock_error_message}"
            )
        if len(self.latencies_ns) != self.iterations:
            validity_errors.append(
                f"collected {len(self.latencies_ns)} of {self.iterations} requested inference samples"
            )
        if dropped_frames:
            validity_errors.append(f"{dropped_frames} synthetic camera frame(s) were dropped")
        validity_errors.extend(condition_warnings)
        valid = not validity_errors
        active_providers = self.session.get_providers()
        record = {
            "model_name": self.model_path.stem,
            "dataset": str(self.get_parameter("dataset").value),
            "device_type": device_type,
            "board": device_type,
            "architecture": platform.machine(),
            "os_version": platform.platform(),
            "valid": valid,
            "profile_version": 3,
            "emulator": False,
            "synthetic_input": self.synthetic_input,
            "input_source": "synthetic_camera" if self.synthetic_input else "sensor_topic",
            "camera_rate_hz": self.camera_rate_hz if self.synthetic_input else None,
            "observed_input_rate_hz": (
                1_000_000_000 / float(np.mean(np.diff(self.arrival_timestamps_ns)))
                if len(self.arrival_timestamps_ns) > 1
                else None
            ),
            "iterations": self.iterations,
            "startup_drop_frames": self.startup_drop_frames,
            "startup_frames_dropped": self.startup_frames_dropped,
            "readiness_handshake": True,
            "image_subscription_queue_depth": 1,
            "failed_frames": self.failed_frames,
            "clock_inconsistency_frames": self.clock_inconsistency_frames,
            "clock_error_message": self.clock_error_message,
            "dropped_frames": dropped_frames,
            "last_frame_error": self.last_frame_error,
            "error_message": "; ".join(validity_errors) or None,
            "duration": int(round(float(np.mean(self.latencies_ns)))),
            "deadline_ms": deadline_ms,
            "deadline_missed_frames": deadline_missed_frames,
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
                active_providers[0]
                if len(active_providers) == 1
                else None
            ),
            "onnxruntime_active_providers": active_providers,
            "onnxruntime_provider_status": (
                "active session providers; per-node execution provider not measured"
            ),
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
            "inference_time_scope": "image_preprocessing_through_output_publish",
            "onnxruntime_inference_time_scope": "session_run_only",
            "model_size_bytes": self.model_path.stat().st_size,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "latency_ns": timing_stats(self.latencies_ns, deadline_ns=deadline_ns),
            "onnxruntime_latency_ns": timing_stats(self.onnxruntime_latencies_ns),
            "end_to_end_latency_ns": (
                timing_stats(self.end_to_end_latencies_ns)
                if self.end_to_end_latencies_ns
                else None
            ),
            "end_to_end_latency_scope": "image_header_stamp_to_output_publish",
            "end_to_end_header_timestamp_frames": self.end_to_end_timestamp_frames,
            "callback_latency_ns": timing_stats(self.callback_latencies_ns),
            "callback_latency_scope": "callback_entry_to_output_publish",
            "source_to_callback_latency_ns": (
                timing_stats(self.source_to_callback_latencies_ns)
                if self.source_to_callback_latencies_ns
                else None
            ),
            "source_to_callback_latency_scope": "image_header_stamp_to_callback_entry",
            "measurement_condition_warnings": condition_warnings,
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
                "cpu_frequencies": cpu_frequencies,
                "temperatures": temperatures,
                "onnxruntime_version": ort.__version__,
                "onnx_num_threads": int(self.get_parameter("onnx_num_threads").value) or None,
                "providers": self.session.get_providers(),
            },
        }
        output_dir = self.results_dir / self.model_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = str(self.get_parameter("dataset").value)
        output_path = output_dir / f"robot_{dataset}_{socket.gethostname()}.json"
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(output_path)
        self.get_logger().info(f"Saved {self.iterations}-frame benchmark to {output_path}")

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