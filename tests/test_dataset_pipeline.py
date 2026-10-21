import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import socket

from ab.bot.dataset_pipeline import _inference_command, _merge_profile, infer_shapes, parser


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

	def test_pipeline_uses_current_python_for_ros_node(self):
		self.assertEqual(_inference_command()[1:], ["-m", "ab.bot.port", "--ros-args"])

	def test_merge_profile_preserves_onnx_timing_and_adds_backend_metrics(self):
		with TemporaryDirectory() as directory:
			args = SimpleNamespace(results_dir=Path(directory), dataset="cifar-10")
			path = args.results_dir / "AirNet" / f"robot_{socket.gethostname()}.json"
			path.parent.mkdir(parents=True)
			path.write_text(json.dumps({
				"valid": True,
				"dataset": "cifar-10",
				"latency_ns": {"avg": 123.0},
				"device_analytics": {
					"timestamp": "ort-time",
					"onnxruntime_version": "test-ort",
					"providers": ["CPUExecutionProvider"],
				},
			}))
			profile = {
				"cpu_duration": 200, "cpu_min_duration": 190, "cpu_max_duration": 210,
				"cpu_std_dev": 10.0, "cpu_runs": 20, "cpu_error": None,
				"gpu_duration": 100, "gpu_min_duration": 95, "gpu_max_duration": 105,
				"gpu_std_dev": 5.0, "gpu_runs": 20, "gpu_error": None,
				"duration": 100, "unit": "Test GPU", "iterations": 20,
				"warmup_runs": 10, "torch_cpu_threads": 4, "torch_gpu_name": "Test GPU",
				"cpu_finite": True, "gpu_finite": True, "cpu_gpu_match": True,
				"cpu_gpu_max_abs_error": 0.0, "cpu_gpu_top1_match": True,
				"cpu_gpu_rtol": 1e-2, "cpu_gpu_atol": 2e-3,
				"pytorch_onnx_match": True, "pytorch_onnx_max_abs_error": 0.0,
				"total_ram_kb": 100, "free_ram_kb": 40, "available_ram_kb": 50,
				"cached_kb": 10, "in_dim_0": 1, "in_dim_1": 3,
				"in_dim_2": 32, "in_dim_3": 32, "torch_version": "test-torch",
				"task": "img-classification", "metric": "accuracy",
				"training_accuracy": 0.9, "checkpoint_repository": "test-repo",
				"checkpoint_epoch": 50,
				"device_analytics": {"cpu_info": {}, "gpu_info": {}, "timeline": [], "profile": {}},
			}
			self.assertTrue(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertEqual(record["onnxruntime_latency_ns"], {"avg": 123.0})
			self.assertEqual(record["cpu_duration"], 200)
			self.assertEqual(record["gpu_duration"], 100)
			self.assertEqual(record["device_analytics"]["onnxruntime_timestamp"], "ort-time")
			self.assertTrue(record["valid"])


if __name__ == "__main__":
	unittest.main()