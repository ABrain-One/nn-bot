import unittest
import json
import csv
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import socket

import numpy as np

from ab.bot.dataset_pipeline import (
	_build_preprocessing_manifest,
	_camera_emulator_command,
	_inference_command,
	_inference_failure_message,
	_has_complete_record,
	_merge_profile,
	_replace_adaptive_avg_pools,
	_write_nn_stat_exports,
	assess_gpu_load,
	infer_shapes,
	parser,
)
from ab.bot.runtime import preprocess_image


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
		self.assertEqual((args.warmup_runs, args.iterations), (10, 500))
		self.assertEqual(args.deadline_ms, 33.0)
		self.assertIsNone(args.cpu_affinity)
		self.assertEqual(args.onnx_num_threads, 1)
		self.assertEqual(args.camera_data_timeout, 90.0)
		self.assertEqual(args.output_normalization, "none")
		self.assertEqual(
			parser().parse_args(
				["--model", "AirNet", "--output-normalization", "l2"]
			).output_normalization,
			"l2",
		)
		self.assertEqual(
			parser().parse_args(
				["--model", "AirNet", "--camera-data-timeout", "45"]
			).camera_data_timeout,
			45.0,
		)

	def test_preprocessing_manifest_matches_nn_dataset_norm_transform(self):
		try:
			from PIL import Image
			transform_module = importlib.import_module("ab.nn.transform.norm_128")
		except (ImportError, RuntimeError) as error:
			self.skipTest(str(error))

		model = {
			"nn": "TestNet",
			"dataset": "cifar-10",
			"prm": {"transform": "norm_128"},
		}
		manifest = _build_preprocessing_manifest(model)
		image = np.random.default_rng(42).integers(0, 256, (40, 60, 3), dtype=np.uint8)
		reference = transform_module.transform(
			(
				(0.4914, 0.4822, 0.4465),
				(0.2470, 0.2435, 0.2616),
			)
		)(Image.fromarray(image)).numpy()[None, ...]

		actual = preprocess_image(
			image,
			(1, 3, 128, 128),
			operations=manifest["operations"],
		)

		np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-6)

	def test_complete_record_matches_preprocessing_and_output_contract(self):
		with TemporaryDirectory() as directory:
			root = Path(directory)
			args = SimpleNamespace(
				results_dir=root / "stats",
				dataset="cifar-10",
				onnx_dir=root / "onnx",
				output_normalization="none",
			)
			manifest = {
				"version": 1,
				"transform": "norm_128",
				"operations": [{"type": "to_tensor", "scale": 1 / 255}],
			}
			manifest_path = args.onnx_dir / args.dataset / "AirNet.preprocess.json"
			manifest_path.parent.mkdir(parents=True)
			manifest_path.write_text(json.dumps(manifest))
			report_path = (
				args.results_dir / "AirNet"
				/ f"robot_{args.dataset}_{socket.gethostname()}.json"
			)
			report_path.parent.mkdir(parents=True)
			report_path.write_text(json.dumps({
				"valid": True,
				"dataset": args.dataset,
				"profile_version": 2,
				"preprocessing": manifest,
				"output_normalization": "none",
			}))

			self.assertTrue(_has_complete_record("AirNet", args))
			args.output_normalization = "l2"
			self.assertFalse(_has_complete_record("AirNet", args))

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

	def test_camera_timeout_has_actionable_failure_message(self):
		from ab.bot.runtime import CAMERA_DATA_TIMEOUT_EXIT_CODE

		self.assertEqual(
			_inference_failure_message(CAMERA_DATA_TIMEOUT_EXIT_CODE, "/camera/raw"),
			"No images received on /camera/raw. Start a camera publisher "
			"or use --emulate-camera.",
		)
		self.assertIsNone(_inference_failure_message(0, "/camera/raw"))

	def test_adaptive_average_pool_exports_and_matches_torch(self):
		try:
			import onnxruntime
			import torch
		except ImportError as error:
			self.skipTest(str(error))

		model = torch.nn.Sequential(
			torch.nn.AdaptiveAvgPool2d((6, 6)),
			torch.nn.Flatten(),
			torch.nn.Linear(3 * 6 * 6, 2),
		).eval()
		example = torch.randn(1, 3, 8, 8)
		expected = model(example).detach().numpy()
		_replace_adaptive_avg_pools(model, torch)

		with TemporaryDirectory() as directory:
			model_path = Path(directory) / "adaptive_pool.onnx"
			torch.onnx.export(
				model,
				example,
				str(model_path),
				opset_version=17,
				input_names=["input"],
				output_names=["output"],
				dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
				dynamo=False,
			)
			session = onnxruntime.InferenceSession(
				str(model_path), providers=["CPUExecutionProvider"]
			)
			actual = session.run(None, {"input": example.numpy()})[0]

		self.assertTrue(
			torch.allclose(
				torch.from_numpy(actual), torch.from_numpy(expected), atol=1e-6
			)
		)

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
				"device_analytics": {
					"timestamp": "ort-time",
					"onnxruntime_version": "test-ort",
					"providers": ["CPUExecutionProvider"],
				},
			}))
			profile = {
				"cpu_duration": 200, "cpu_min_duration": 190, "cpu_max_duration": 210,
				"cpu_median_duration": 200, "cpu_p95_duration": 209, "cpu_p99_duration": 209,
				"cpu_std_dev": 10.0, "cpu_runs": 20, "cpu_error": None,
				"cpu_runs_ns": [200] * 20,
				"gpu_duration": 100, "gpu_min_duration": 95, "gpu_max_duration": 105,
				"gpu_median_duration": 100, "gpu_p95_duration": 105, "gpu_p99_duration": 105,
				"gpu_std_dev": 5.0, "gpu_runs": 20, "gpu_error": None,
				"gpu_runs_ns": [100] * 20,
				"npu_duration": None, "npu_min_duration": None, "npu_max_duration": None,
				"npu_std_dev": None, "npu_runs": 0, "npu_error": None,
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
			self.assertNotIn("onnxruntime_latency_ns", record)
			self.assertEqual(record["latency_ns"], {"avg": 456.0})
			self.assertEqual(record["cpu_duration"], 200)
			self.assertEqual(record["cpu_p99_duration"], 209)
			self.assertEqual(record["gpu_duration"], 100)
			self.assertEqual(record["gpu_p95_duration"], 105)
			self.assertEqual(record["gpu_p99_duration"], 105)
			self.assertIsNone(record["npu_duration"])
			self.assertEqual(record["npu_runs"], 0)
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
			self.assertEqual(
				record["nn_stat"],
				{
					"task": "img-classification",
					"dataset": "cifar-10",
					"epoch": 50,
					"metric": "acc",
					"accuracy": 0.9,
					"duration": 456 / 1_000_000_000,
					"nn": "AirNet",
					"accuracy_source": "checkpoint_training_accuracy",
					"duration_unit": "seconds_per_inference",
					"duration_source": "image_preprocessing_through_output_publish",
				},
			)
			nn_stat_csv = path.with_name(f"{path.stem}_nn_stat.csv")
			with nn_stat_csv.open(newline="", encoding="utf-8") as csv_file:
				nn_stat_row = next(csv.DictReader(csv_file))
			self.assertEqual(
				{key: nn_stat_row[key] for key in (
					"task", "dataset", "epoch", "metric", "accuracy", "nn"
				)},
				{
					"task": "img-classification",
					"dataset": "cifar-10",
					"epoch": "50",
					"metric": "acc",
					"accuracy": "0.9",
					"nn": "AirNet",
				},
			)
			self.assertTrue(record["valid"])

			record.pop("nn_stat")
			path.write_text(json.dumps(record))
			nn_stat_csv.unlink()
			_write_nn_stat_exports(path, "AirNet")
			self.assertIn("nn_stat", json.loads(path.read_text()))
			self.assertTrue(nn_stat_csv.is_file())

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
				"clock_inconsistency_frames": 0,
				"last_frame_error": "RuntimeError: dropped frame",
				"latency_ns": {"avg": 123.0},
				"device_analytics": {"providers": ["CPUExecutionProvider"]},
			}))
			profile = {
				"cpu_duration": 200, "cpu_min_duration": 190, "cpu_max_duration": 210,
				"cpu_median_duration": 200, "cpu_p95_duration": 209, "cpu_p99_duration": 209,
				"cpu_std_dev": 10.0, "cpu_runs": 20, "cpu_error": None,
				"cpu_runs_ns": [200] * 20,
				"gpu_duration": None, "gpu_min_duration": None, "gpu_max_duration": None,
				"gpu_median_duration": None, "gpu_p95_duration": None, "gpu_p99_duration": None,
				"gpu_std_dev": None, "gpu_runs": 0, "gpu_error": "CUDA is not available",
				"gpu_runs_ns": [],
				"npu_duration": None, "npu_min_duration": None, "npu_max_duration": None,
				"npu_std_dev": None, "npu_runs": 0, "npu_error": None,
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
			self.assertEqual(
				record["inference_time_scope"],
				"image_preprocessing_through_output_publish",
			)
			self.assertIn("CUDA is not available", record["error_message"])
			self.assertIn("2 image frame(s) failed", record["error_message"])
			self.assertEqual(record["failed_frames"], 2)

			record["failed_frames"] = 0
			record["valid"] = True
			record["clock_inconsistency_frames"] = 1
			path.write_text(json.dumps(record))
			self.assertFalse(_merge_profile("AirNet", args, profile))
			record = json.loads(path.read_text())
			self.assertFalse(record["valid"])
			self.assertIn("1 image frame(s) had inconsistent ROS clocks", record["error_message"])

			record["clock_inconsistency_frames"] = 0
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