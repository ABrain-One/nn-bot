import unittest
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import socket

from ab.bot.dataset_pipeline import (
	_camera_emulator_command,
	_inference_command,
	_merge_profile,
	assess_gpu_load,
	infer_shapes,
	parser,
)


class DatasetPipelineTests(unittest.TestCase):
	def test_infers_standard_image_shapes(self):
		self.assertEqual(infer_shapes("cifar-100", ""), ((1, 3, 32, 32), (100,)))
		self.assertEqual(infer_shapes("mnist", ""), ((1, 1, 28, 28), (10,)))
		self.assertEqual(infer_shapes("cifar-10", "echo_128"), ((1, 3, 128, 128), (10,)))

	def test_cli_requires_one_model_selection(self):
		with self.assertRaises(SystemExit):
			parser().parse_args([])

	def test_cli_accepts_one_model_or_all_models(self):
		self.assertEqual(parser().parse_args(["--model", "AirNet"]).model, "AirNet")
		self.assertTrue(parser().parse_args(["--all-models"]).all_models)

	def test_cli_saves_results_under_stats_by_default(self):
		self.assertEqual(parser().parse_args(["--model", "AirNet"]).results_dir, Path("stats"))
		args = parser().parse_args(["--model", "AirNet"])
		self.assertEqual((args.warmup_runs, args.iterations), (10, 20))

	def test_cli_supports_export_only(self):
		self.assertTrue(parser().parse_args(["--model", "AirNet", "--export-only"]).export_only)

	def test_cli_configures_gpu_load_policy(self):
		defaults = parser().parse_args(["--model", "AirNet"])
		strict = parser().parse_args(["--model", "AirNet", "--gpu-busy-policy", "fail"])

		self.assertEqual(defaults.gpu_busy_policy, "warn")
		self.assertEqual(defaults.gpu_utilization_limit, 20.0)
		self.assertEqual(defaults.gpu_memory_limit_percent, 20.0)
		self.assertEqual(strict.gpu_busy_policy, "fail")

	def test_cli_supports_camera_emulation_and_rate(self):
		args = parser().parse_args(["--model", "AirNet", "--emulate-camera", "--camera-rate-hz", "24"])

		self.assertTrue(args.emulate_camera)
		self.assertEqual(args.camera_rate_hz, 24.0)
		self.assertEqual(
			_camera_emulator_command(args),
			[
				sys.executable,
				"-m",
				"ab.bot.synthetic_camera",
				"--ros-args",
				"-p",
				"image_topic:=/camera/image_raw",
				"-p",
				"publish_rate_hz:=24.0",
			],
		)

	def test_gpu_preflight_detects_utilization_and_memory_contention(self):
		assessment = assess_gpu_load(
			[{
				"index": 0,
				"name": "Test GPU",
				"memory_total_mb": 10000,
				"memory_used_mb": 5000,
				"util_percent": 93,
			}]
		)

		self.assertTrue(assessment["available"])
		self.assertTrue(assessment["busy"])
		self.assertEqual(assessment["devices"][0]["memory_used_percent"], 50.0)
		self.assertEqual(len(assessment["warnings"]), 2)

	def test_gpu_preflight_reports_unavailable_sampler(self):
		assessment = assess_gpu_load(None)

		self.assertFalse(assessment["available"])
		self.assertFalse(assessment["busy"])
		self.assertTrue(assessment["warnings"])

	def test_pipeline_uses_current_python_for_ros_node(self):
		self.assertEqual(_inference_command()[1:], ["-m", "ab.bot.port", "--ros-args"])

	def test_merge_profile_preserves_onnx_timing_and_adds_backend_metrics(self):
		with TemporaryDirectory() as directory:
			args = SimpleNamespace(results_dir=Path(directory), dataset="cifar-10")
			path = args.results_dir / "AirNet" / f"robot_cifar-10_{socket.gethostname()}.json"
			path.parent.mkdir(parents=True)
			path.write_text(json.dumps({
				"valid": True,
				"dataset": "cifar-10",
				"duration": 456,
				"unit": "Robot Hardware",
				"inference_time_scope": "image_preprocessing_through_output_publish",
				"latency_ns": {"avg": 456.0},
				"onnxruntime_latency_ns": {"avg": 123.0},
				"device_analytics": {
					"timestamp": "ort-time",
					"onnxruntime_version": "test-ort",
					"providers": ["CPUExecutionProvider"],
				},
			}))
			profile = {
				"cpu_duration": 200, "cpu_min_duration": 190, "cpu_max_duration": 210,
				"cpu_median_duration": 200, "cpu_p95_duration": 209,
				"cpu_std_dev": 10.0, "cpu_runs": 20, "cpu_error": None,
				"cpu_runs_ns": [200] * 20,
				"gpu_duration": 100, "gpu_min_duration": 95, "gpu_max_duration": 105,
				"gpu_median_duration": 100, "gpu_p95_duration": 105,
				"gpu_std_dev": 5.0, "gpu_runs": 20, "gpu_error": None,
				"gpu_runs_ns": [100] * 20,
				"duration": 100, "unit": "Test GPU", "iterations": 20,
				"warmup_runs": 10, "torch_cpu_threads": 4, "torch_gpu_name": "Test GPU",
				"cpu_finite": True, "gpu_finite": True, "cpu_gpu_match": True,
				"cpu_gpu_max_abs_error": 0.0, "cpu_gpu_top1_match": True,
				"cpu_gpu_rtol": 1e-2, "cpu_gpu_atol": 2e-3,
				"pytorch_onnx_match": True, "pytorch_onnx_max_abs_error": 0.0,
				"gpu_onnx_match": True, "gpu_onnx_max_abs_error": 0.0,
				"total_ram_kb": 100, "free_ram_kb": 40, "available_ram_kb": 50,
				"cached_kb": 10, "in_dim_0": 1, "in_dim_1": 3,
				"in_dim_2": 32, "in_dim_3": 32, "torch_version": "test-torch",
				"task": "img-classification", "metric": "accuracy",
				"training_accuracy": 0.9, "checkpoint_repository": "test-repo",
				"checkpoint_epoch": 50,
				"device_analytics": {"cpu_info": {}, "gpu_info": {}, "timeline": [], "profile": {}},
				"onnxruntime_consistency_providers": ["CPUExecutionProvider"],
				"gpu_preflight_snapshot": [{"index": 0, "util_percent": 2}],
				"gpu_preflight": {"available": True, "busy": False, "warnings": []},
			}
			self.assertTrue(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertEqual(record["onnxruntime_latency_ns"], {"avg": 123.0})
			self.assertEqual(record["cpu_duration"], 200)
			self.assertEqual(record["gpu_duration"], 100)
			self.assertEqual(record["gpu_p95_duration"], 105)
			self.assertEqual(record["duration"], 456)
			self.assertEqual(record["unit"], "Robot Hardware")
			self.assertEqual(
				record["inference_time_scope"],
				"image_preprocessing_through_output_publish",
			)
			self.assertEqual(record["pytorch_duration"], 100)
			self.assertEqual(record["pytorch_unit"], "Test GPU")
			self.assertEqual(
				record["pytorch_inference_time_scope"], "pytorch_cuda_forward_only"
			)
			self.assertEqual(record["onnxruntime_provider"], "CPUExecutionProvider")
			self.assertEqual(
				record["onnxruntime_consistency_providers"],
				["CPUExecutionProvider"],
			)
			self.assertEqual(len(record["gpu_runs_ns"]), 20)
			self.assertEqual(
				record["gpu_preflight"]["assessment"]["busy"],
				False,
			)
			self.assertEqual(record["device_analytics"]["onnxruntime_timestamp"], "ort-time")
			self.assertTrue(record["valid"])

			profile["cpu_finite"] = False
			profile["pytorch_onnx_match"] = False
			self.assertTrue(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertTrue(record["valid"])
			self.assertEqual(record["duration"], 456)
			self.assertEqual(record["unit"], "Robot Hardware")
			self.assertEqual(record["pytorch_duration"], 100)
			self.assertEqual(record["pytorch_unit"], "Test GPU")

	def test_merge_profile_accepts_correct_cpu_without_cuda(self):
		with TemporaryDirectory() as directory:
			args = SimpleNamespace(results_dir=Path(directory), dataset="cifar-10")
			path = args.results_dir / "AirNet" / f"robot_cifar-10_{socket.gethostname()}.json"
			path.parent.mkdir(parents=True)
			path.write_text(json.dumps({
				"valid": True,
				"dataset": "cifar-10",
				"duration": 123,
				"unit": "Robot Hardware",
				"inference_time_scope": "image_preprocessing_through_output_publish",
				"failed_frames": 2,
				"last_frame_error": "RuntimeError: dropped frame",
				"latency_ns": {"avg": 123.0},
				"device_analytics": {"providers": ["CPUExecutionProvider"]},
			}))
			profile = {
				"cpu_duration": 200, "cpu_min_duration": 190, "cpu_max_duration": 210,
				"cpu_median_duration": 200, "cpu_p95_duration": 209,
				"cpu_std_dev": 10.0, "cpu_runs": 20, "cpu_error": None,
				"cpu_runs_ns": [200] * 20,
				"gpu_duration": None, "gpu_min_duration": None, "gpu_max_duration": None,
				"gpu_median_duration": None, "gpu_p95_duration": None,
				"gpu_std_dev": None, "gpu_runs": 0, "gpu_error": "CUDA is not available",
				"gpu_runs_ns": [],
				"duration": 200, "unit": "CPU", "iterations": 20,
				"warmup_runs": 10, "torch_cpu_threads": 4, "torch_gpu_name": None,
				"cpu_finite": True, "gpu_finite": None, "cpu_gpu_match": None,
				"cpu_gpu_max_abs_error": None, "cpu_gpu_top1_match": None,
				"cpu_gpu_rtol": 1e-2, "cpu_gpu_atol": 2e-3,
				"pytorch_onnx_match": True, "pytorch_onnx_max_abs_error": 0.0,
				"gpu_onnx_match": None, "gpu_onnx_max_abs_error": None,
				"total_ram_kb": 100, "free_ram_kb": 40, "available_ram_kb": 50,
				"cached_kb": 10, "in_dim_0": 1, "in_dim_1": 3,
				"in_dim_2": 32, "in_dim_3": 32, "torch_version": "test-torch",
				"task": "img-classification", "metric": "accuracy",
				"training_accuracy": 0.9, "checkpoint_repository": "test-repo",
				"checkpoint_epoch": 50,
				"device_analytics": {"cpu_info": {}, "gpu_info": {}, "timeline": [], "profile": {}},
				"onnxruntime_consistency_providers": ["CPUExecutionProvider"],
				"gpu_preflight_snapshot": None,
				"gpu_preflight": {"available": False, "busy": False, "warnings": []},
			}
			self.assertFalse(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertFalse(record["valid"])
			self.assertEqual(record["inference_time_scope"], "image_preprocessing_through_output_publish")
			self.assertIn("CUDA is not available", record["error_message"])
			self.assertIn("2 image frame(s) failed", record["error_message"])
			self.assertEqual(record["failed_frames"], 2)

			record["failed_frames"] = 0
			record["valid"] = True
			path.write_text(json.dumps(record))
			self.assertTrue(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertTrue(record["valid"])
			self.assertEqual(record["duration"], 123)
			self.assertEqual(record["unit"], "Robot Hardware")
			self.assertEqual(record["pytorch_duration"], 200)
			self.assertEqual(record["pytorch_unit"], "CPU")
			self.assertEqual(
				record["pytorch_inference_time_scope"], "pytorch_cpu_forward_only"
			)
			self.assertIsNone(record["error_message"])


if __name__ == "__main__":
	unittest.main()