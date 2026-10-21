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
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ab.bot.runtime import device_name, preprocess_image, timing_stats


class InferenceNode(Node):
	def __init__(self):
		super().__init__("nn_bot_inference")
		self.declare_parameter("model_path", "")
		self.declare_parameter("input_topic", "/camera/image_raw")
		self.declare_parameter("output_topic", "/nn_bot/output")
		self.declare_parameter("results_dir", "stats")
		self.declare_parameter("dataset", "custom")
		self.declare_parameter("warmup_runs", 10)
		self.declare_parameter("iterations", 20)
		self.declare_parameter("input_scale", 1.0 / 255.0)
		self.declare_parameter("input_mean", Parameter.Type.DOUBLE_ARRAY)
		self.declare_parameter("input_std", Parameter.Type.DOUBLE_ARRAY)
		self.declare_parameter("providers", ["CPUExecutionProvider"])
		self.declare_parameter("exit_after_benchmark", False)

		model_path = Path(self.get_parameter("model_path").value).expanduser()
		if not model_path.is_file():
			raise FileNotFoundError(f"ONNX model not found: {model_path}")

		requested_providers = list(self.get_parameter("providers").value)
		available_providers = ort.get_available_providers()
		providers = [name for name in requested_providers if name in available_providers]
		if not providers:
			raise RuntimeError(
				f"None of the requested ONNX Runtime providers are available: "
				f"{requested_providers}; available: {available_providers}"
			)

		self.session = ort.InferenceSession(str(model_path), providers=providers)
		self.model_input = self.session.get_inputs()[0]
		self.model_path = model_path
		self.input_scale = float(self.get_parameter("input_scale").value)
		self.input_mean = list(self.get_parameter("input_mean").value)
		self.input_std = list(self.get_parameter("input_std").value)
		self.warmup_runs = max(0, int(self.get_parameter("warmup_runs").value))
		self.iterations = int(self.get_parameter("iterations").value)
		if self.iterations < 1:
			raise ValueError("iterations must be at least 1")

		self.results_dir = Path(self.get_parameter("results_dir").value).expanduser()
		self.bridge = CvBridge()
		self.completed_warmups = 0
		self.latencies_ns = []
		self.record_saved = False
		self.shutdown_timer = None
		self.output_publisher = self.create_publisher(
			Float32MultiArray, str(self.get_parameter("output_topic").value), 10
		)
		self.subscription = self.create_subscription(
			Image,
			str(self.get_parameter("input_topic").value),
			self.on_image,
			10,
		)
		self.get_logger().info(
			f"Loaded {model_path.name}; providers: {', '.join(self.session.get_providers())}"
		)

	def on_image(self, message):
		try:
			image = self.bridge.imgmsg_to_cv2(message, desired_encoding="rgb8")
			model_tensor = preprocess_image(
				image,
				self.model_input.shape,
				self.input_scale,
				self.input_mean,
				self.input_std,
			)
			started = time.perf_counter_ns()
			outputs = self.session.run(None, {self.model_input.name: model_tensor})
			elapsed_ns = time.perf_counter_ns() - started
			output = np.asarray(outputs[0], dtype=np.float32)

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

			if self.completed_warmups < self.warmup_runs:
				self.completed_warmups += 1
				return

			if len(self.latencies_ns) < self.iterations:
				self.latencies_ns.append(elapsed_ns)
				if len(self.latencies_ns) == self.iterations and not self.record_saved:
					self.save_record(list(model_tensor.shape), list(output.shape))
					self.record_saved = True
					if self.get_parameter("exit_after_benchmark").value:
						self.shutdown_timer = self.create_timer(0.1, self.stop_after_benchmark)
		except Exception as error:
			self.get_logger().error(f"Inference failed: {error}")

	def save_record(self, input_shape, output_shape):
		device_type = device_name()
		record = {
			"model_name": self.model_path.stem,
			"dataset": str(self.get_parameter("dataset").value),
			"device_type": device_type,
			"os_version": platform.platform(),
			"valid": True,
			"emulator": False,
			"iterations": self.iterations,
			"error_message": None,
			"duration": int(round(float(np.mean(self.latencies_ns)))),
			"unit": self.session.get_providers()[0],
			"runtime": "ONNX Runtime",
			"onnxruntime_provider": self.session.get_providers()[0],
			"model_format": "onnx",
			"input_topic": str(self.get_parameter("input_topic").value),
			"inference_time_scope": "onnxruntime_session.run_only",
			"model_size_bytes": self.model_path.stat().st_size,
			"input_shape": input_shape,
			"output_shape": output_shape,
			"latency_ns": timing_stats(self.latencies_ns),
			"onnxruntime_latency_ns": timing_stats(self.latencies_ns),
			"in_dim_0": input_shape[0],
			"in_dim_1": input_shape[1],
			"in_dim_2": input_shape[2],
			"in_dim_3": input_shape[3],
			"device_analytics": {
				"timestamp": datetime.now(timezone.utc).isoformat(),
				"python_version": platform.python_version(),
				"ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
				"cpu_cores": os.cpu_count(),
				"onnxruntime_version": ort.__version__,
				"providers": self.session.get_providers(),
			},
		}
		output_dir = self.results_dir / self.model_path.stem
		output_dir.mkdir(parents=True, exist_ok=True)
		output_path = output_dir / f"robot_{socket.gethostname()}.json"
		temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
		temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
		temporary_path.replace(output_path)
		self.get_logger().info(f"Saved {self.iterations}-frame benchmark to {output_path}")

	def stop_after_benchmark(self):
		self.shutdown_timer.cancel()
		rclpy.try_shutdown()


def main(args=None):
	rclpy.init(args=args)
	node = None
	executor = SingleThreadedExecutor()
	try:
		node = InferenceNode()
		executor.add_node(node)
		while rclpy.ok():
			executor.spin_once(timeout_sec=0.1)
	finally:
		if node is not None:
			executor.remove_node(node)
			node.destroy_node()
		executor.shutdown()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()
