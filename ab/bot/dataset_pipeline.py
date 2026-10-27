"""Export and benchmark NN-Dataset image classifiers on a ROS 2 robot."""

import argparse
import ast
import datetime
import gc
import importlib
import inspect
import json
import math
import os
import platform
import pkgutil
import re
import signal
import socket
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from ab.bot.runtime import sample_gpus, sample_system, timing_stats


CHECKPOINT_REPOSITORY = "NN-Dataset/checkpoints-epoch-50"


def infer_shapes(dataset, transform):
    match = re.search(r"(?:echo|norm)_(\d+)", transform or "")
    if not match:
        match = re.search(r"\b(64|96|128|160|192|224|256|288|299|331|384|512)\b", transform or "")

    if match:
        size = int(match.group(1))
        input_shape = (1, 3, size, size)
    elif dataset == "mnist":
        input_shape = (1, 1, 28, 28)
    elif dataset == "imagenette":
        input_shape = (1, 3, 160, 160)
    else:
        input_shape = (1, 3, 32, 32)

    return input_shape, (100 if dataset == "cifar-100" else 10,)


def _load_dataset_dependencies():
    try:
        import torch
        import onnx
        from huggingface_hub import hf_hub_download, list_repo_files
        import ab.nn.nn as architectures
    except ImportError as error:
        raise RuntimeError(
            "Dataset runs need NN-Dataset, PyTorch, ONNX, and huggingface_hub. "
            "Install requirements-dataset.txt and NN-Dataset first."
        ) from error
    return torch, onnx, hf_hub_download, list_repo_files, architectures


def discover_models(dataset, selected_model=None):
    _, _, hf_hub_download, list_repo_files, architectures = _load_dataset_dependencies()
    architecture_names = {
        module.name
        for module in pkgutil.iter_modules(architectures.__path__)
        if not module.ispkg
    }
    checkpoint_names = {
        Path(name).stem
        for name in list_repo_files(CHECKPOINT_REPOSITORY)
        if name.endswith(".pth") and "/" not in name
    }
    available = sorted(architecture_names & checkpoint_names)
    if selected_model is not None:
        if selected_model not in architecture_names:
            raise ValueError(f"No NN-Dataset architecture module named {selected_model!r}")
        if selected_model not in checkpoint_names:
            raise ValueError(f"No epoch-50 checkpoint for {selected_model!r}")
        available = [selected_model]
    if not available:
        raise RuntimeError("No NN-Dataset architectures have matching epoch-50 checkpoints")

    cache_dir = Path("_work") / "huggingface"
    metadata_path = hf_hub_download(
        repo_id=CHECKPOINT_REPOSITORY,
        filename="all_models.json",
        cache_dir=str(cache_dir),
    )
    with open(metadata_path, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    models = []
    for name in available:
        entry = metadata.get(name, {}) or {}
        prm = entry.get("prm", {}) if isinstance(entry, dict) else {}
        if isinstance(prm, str):
            try:
                prm = ast.literal_eval(prm)
            except (ValueError, SyntaxError):
                prm = {}
        models.append(
            {
                "nn": name,
                "task": "img-classification",
                "dataset": dataset,
                "metric": "accuracy",
                "training_accuracy": entry.get("accuracy") if isinstance(entry, dict) else None,
                "checkpoint_repository": CHECKPOINT_REPOSITORY,
                "checkpoint_epoch": 50,
                "prm": prm or {},
            }
        )
    return models, hf_hub_download


def _load_torch_model(model_row, hf_hub_download, device):
    import torch

    model_name = model_row["nn"]
    dataset = model_row["dataset"]
    prm = model_row["prm"]
    input_shape, output_shape = infer_shapes(dataset, prm.get("transform", ""))
    module = importlib.import_module(f"ab.nn.nn.{model_name}")
    if not hasattr(module, "Net"):
        raise RuntimeError(f"NN-Dataset module {model_name} does not define Net")

    model = module.Net(input_shape, output_shape, prm, device)
    checkpoint_path = hf_hub_download(
        repo_id=CHECKPOINT_REPOSITORY,
        filename=f"{model_name}.pth",
        cache_dir=str(Path("_work") / "huggingface"),
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    if hasattr(model, "device"):
        model.device = device
    return torch, model, input_shape


def export_model(model_row, output_path, hf_hub_download):
    import torch
    import onnx

    output_path = Path(output_path)
    torch, model, input_shape = _load_torch_model(model_row, hf_hub_download, torch.device("cpu"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_options = {
        "opset_version": 17,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {"input": {0: "batch"}, "output": {0: "batch"}},
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(model, torch.randn(input_shape), str(output_path), **export_options)
    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


def _snapshot(phase):
    return {
        "phase": phase,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "sys": sample_system(),
        "gpus": sample_gpus(),
    }


def assess_gpu_load(gpus, utilization_limit=20.0, memory_limit_percent=20.0):
    """Summarize GPU contention against pre-run utilization and memory thresholds."""
    if not gpus:
        return {
            "available": False,
            "busy": False,
            "thresholds": {
                "utilization_percent": utilization_limit,
                "memory_used_percent": memory_limit_percent,
            },
            "devices": [],
            "warnings": ["GPU usage could not be sampled (nvidia-smi unavailable or failed)"],
        }

    devices = []
    warnings = []
    for gpu in gpus:
        memory_total = gpu.get("memory_total_mb")
        memory_used = gpu.get("memory_used_mb")
        memory_used_percent = (
            100.0 * memory_used / memory_total if memory_total and memory_used is not None else None
        )
        device = {
            **gpu,
            "memory_used_percent": memory_used_percent,
            "busy": False,
        }
        if gpu.get("util_percent") is not None and gpu["util_percent"] > utilization_limit:
            device["busy"] = True
            warnings.append(
                f'GPU {gpu.get("index", "?")} utilization is {gpu["util_percent"]}% '
                f"(limit {utilization_limit:g}%)"
            )
        if memory_used_percent is not None and memory_used_percent > memory_limit_percent:
            device["busy"] = True
            warnings.append(
                f'GPU {gpu.get("index", "?")} memory use is {memory_used_percent:.1f}% '
                f"({memory_used} of {memory_total} MB; limit {memory_limit_percent:g}%)"
            )
        devices.append(device)

    return {
        "available": True,
        "busy": any(device["busy"] for device in devices),
        "thresholds": {
            "utilization_percent": utilization_limit,
            "memory_used_percent": memory_limit_percent,
        },
        "devices": devices,
        "warnings": warnings,
    }


def _model_output(output):
    if isinstance(output, dict):
        output = next(iter(output.values()))
    elif isinstance(output, (tuple, list)):
        output = output[0]
    return output


def _dataset_sample(model_row):
    import torch
    from ab.nn.util.Loader import load_dataset
    from ab.nn.util.Util import first_tensor

    task = model_row.get("task", "img-classification")
    dataset = model_row["dataset"]
    prm = model_row["prm"]
    transform = prm.get("transform")
    if not transform:
        raise ValueError(f"NN-Dataset metadata has no transform for {model_row['nn']}")
    num_workers = int(prm.get("num_workers", 1))
    _, _, train_set, _ = load_dataset(task, dataset, transform)
    sample = first_tensor(train_set, num_workers=num_workers)
    if sample is None:
        raise RuntimeError(f"NN-Dataset returned no sample for {task}/{dataset}/{transform}")
    return sample.detach().to(device="cpu", dtype=torch.float32)


def profile_model(
    model_row,
    onnx_path,
    hf_hub_download,
    iterations=20,
    warmup_runs=10,
    input_tensor=None,
    gpu_utilization_limit=20.0,
    gpu_memory_limit_percent=20.0,
    gpu_busy_policy="warn",
):
    """Profile one transformed NN-Dataset sample on PyTorch CPU and CUDA."""
    import torch
    import onnxruntime as ort

    cpu_device = torch.device("cpu")
    torch, cpu_model, input_shape = _load_torch_model(model_row, hf_hub_download, cpu_device)
    if input_tensor is None:
        cpu_input = _dataset_sample(model_row)
        input_source = "first transformed NN-Dataset training sample"
    else:
        cpu_input = input_tensor.detach().to(device="cpu", dtype=torch.float32)
        input_source = "explicit profiler validation tensor"
    if tuple(cpu_input.shape) != tuple(input_shape):
        raise ValueError(
            f"NN-Dataset sample shape {tuple(cpu_input.shape)} does not match model input {input_shape}"
        )
    dataset = model_row["dataset"]

    timeline = [_snapshot("before_eval")]
    cpu_runs = []
    with torch.inference_mode():
        for _ in range(warmup_runs):
            _model_output(cpu_model(cpu_input))
        for _ in range(iterations):
            started = __import__("time").perf_counter_ns()
            cpu_output = _model_output(cpu_model(cpu_input))
            cpu_runs.append(__import__("time").perf_counter_ns() - started)
    cpu_output = cpu_output.detach().cpu()
    cpu_finite = bool(torch.isfinite(cpu_output).all().item())
    after_cpu_snapshot = _snapshot("after_cpu")
    timeline.append(after_cpu_snapshot)
    gpu_preflight_snapshot = after_cpu_snapshot.get("gpus")
    gpu_preflight = assess_gpu_load(
        gpu_preflight_snapshot,
        utilization_limit=gpu_utilization_limit,
        memory_limit_percent=gpu_memory_limit_percent,
    )

    gpu_runs = []
    gpu_error = None
    gpu_output = None
    gpu_name = None
    torch_peak_gpu_allocated_mb = None
    if gpu_preflight["busy"] and gpu_busy_policy == "fail":
        gpu_error = "GPU load exceeded configured limits; CUDA profiling skipped by policy"
    elif torch.cuda.is_available():
        try:
            gpu_device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(gpu_device)
            gpu_model = type(cpu_model)(
                input_shape,
                (100 if dataset == "cifar-100" else 10,),
                model_row["prm"],
                gpu_device,
            )
            gpu_model.load_state_dict(cpu_model.state_dict(), strict=True)
            gpu_model.eval().to(gpu_device)
            if hasattr(gpu_model, "device"):
                gpu_model.device = gpu_device
            gpu_input = cpu_input.to(gpu_device)
            torch.cuda.reset_peak_memory_stats(gpu_device)
            with torch.inference_mode():
                for _ in range(warmup_runs):
                    _model_output(gpu_model(gpu_input))
                torch.cuda.synchronize(gpu_device)
                for _ in range(iterations):
                    started = torch.cuda.Event(enable_timing=True)
                    finished = torch.cuda.Event(enable_timing=True)
                    started.record()
                    gpu_output = _model_output(gpu_model(gpu_input))
                    finished.record()
                    finished.synchronize()
                    gpu_runs.append(int(round(started.elapsed_time(finished) * 1_000_000)))
            gpu_output = gpu_output.detach().cpu()
            torch_peak_gpu_allocated_mb = torch.cuda.max_memory_allocated(gpu_device) / (1024 * 1024)
        except Exception as error:
            gpu_error = f"{type(error).__name__}: {error}"
            gpu_runs = []
            gpu_output = None
        finally:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    else:
        gpu_error = "CUDA is not available to PyTorch"
    timeline.append(_snapshot("after_gpu"))

    cpu_stats = timing_stats(cpu_runs)
    gpu_stats = timing_stats(gpu_runs) if gpu_runs else None
    gpu_finite = bool(torch.isfinite(gpu_output).all().item()) if gpu_output is not None else None
    cpu_gpu_match = None
    cpu_gpu_max_abs_error = None
    cpu_gpu_top1_match = None
    if gpu_output is not None:
        cpu_gpu_max_abs_error = float(torch.max(torch.abs(cpu_output - gpu_output)).item())
        cpu_gpu_match = bool(torch.allclose(cpu_output, gpu_output, rtol=1e-2, atol=2e-3))
        if cpu_output.ndim >= 2 and cpu_output.shape[0] == 1:
            cpu_gpu_top1_match = bool(
                torch.equal(cpu_output.flatten(1).argmax(1), gpu_output.flatten(1).argmax(1))
            )

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"ONNX Runtime CPU provider is unavailable: {ort.get_available_providers()}"
        )
    onnx_input = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = onnx_input.run(
        None,
        {onnx_input.get_inputs()[0].name: cpu_input.detach().cpu().numpy()},
    )[0]
    onnx_max_abs_error = float(np.max(np.abs(cpu_output.numpy() - onnx_output)))
    pytorch_onnx_match = bool(np.allclose(cpu_output.numpy(), onnx_output, rtol=1e-3, atol=1e-4))
    gpu_onnx_max_abs_error = (
        float(np.max(np.abs(gpu_output.numpy() - onnx_output)))
        if gpu_output is not None
        else None
    )
    gpu_onnx_match = (
        bool(np.allclose(gpu_output.numpy(), onnx_output, rtol=1e-3, atol=1e-4))
        if gpu_output is not None
        else None
    )

    successful = {"CPU": cpu_stats["avg"]}
    if gpu_stats:
        successful["GPU"] = gpu_stats["avg"]
    fastest = min(successful, key=successful.get)
    gpu_snapshots = [gpu for item in timeline for gpu in (item.get("gpus") or [])]
    system_snapshots = [item["sys"] for item in timeline if item.get("sys")]
    cpu_util = [item.get("cpu_percent") for item in system_snapshots if item.get("cpu_percent") is not None]
    gpu_util = [item.get("util_percent") for item in gpu_snapshots if item.get("util_percent") is not None]
    gpu_memory = [item.get("memory_used_mb") for item in gpu_snapshots if item.get("memory_used_mb") is not None]
    process_rss = [item.get("process_rss_bytes") for item in system_snapshots if item.get("process_rss_bytes") is not None]
    memory = system_snapshots[-1] if system_snapshots else {}

    profile = {
        "cpu_duration": int(round(cpu_stats["avg"])),
        "cpu_min_duration": int(round(cpu_stats["min"])),
        "cpu_max_duration": int(round(cpu_stats["max"])),
        "cpu_median_duration": int(round(cpu_stats["median"])),
        "cpu_p95_duration": int(round(cpu_stats["p95"])),
        "cpu_std_dev": cpu_stats["std"],
        "cpu_runs": len(cpu_runs),
        "cpu_error": None,
        "gpu_duration": int(round(gpu_stats["avg"])) if gpu_stats else None,
        "gpu_min_duration": int(round(gpu_stats["min"])) if gpu_stats else None,
        "gpu_max_duration": int(round(gpu_stats["max"])) if gpu_stats else None,
        "gpu_median_duration": int(round(gpu_stats["median"])) if gpu_stats else None,
        "gpu_p95_duration": int(round(gpu_stats["p95"])) if gpu_stats else None,
        "gpu_std_dev": gpu_stats["std"] if gpu_stats else None,
        "gpu_runs": len(gpu_runs),
        "gpu_error": gpu_error,
        "cpu_runs_ns": cpu_runs,
        "gpu_runs_ns": gpu_runs,
        "duration": int(round(successful[fastest])),
        "unit": gpu_name if fastest == "GPU" else "CPU",
        "iterations": iterations,
        "warmup_runs": warmup_runs,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_gpu_name": gpu_name,
        "cpu_finite": cpu_finite,
        "gpu_finite": gpu_finite,
        "cpu_gpu_match": cpu_gpu_match,
        "cpu_gpu_max_abs_error": cpu_gpu_max_abs_error,
        "cpu_gpu_top1_match": cpu_gpu_top1_match,
        "cpu_gpu_rtol": 1e-2,
        "cpu_gpu_atol": 2e-3,
        "pytorch_onnx_match": pytorch_onnx_match,
        "pytorch_onnx_max_abs_error": onnx_max_abs_error,
        "gpu_onnx_match": gpu_onnx_match,
        "gpu_onnx_max_abs_error": gpu_onnx_max_abs_error,
        "onnxruntime_consistency_providers": onnx_input.get_providers(),
        "torch_version": torch.__version__,
        "task": model_row.get("task", "img-classification"),
        "metric": model_row.get("metric", "accuracy"),
        "training_accuracy": model_row.get("training_accuracy"),
        "checkpoint_repository": model_row.get("checkpoint_repository", CHECKPOINT_REPOSITORY),
        "checkpoint_epoch": model_row.get("checkpoint_epoch", 50),
        "total_ram_kb": memory.get("total_ram_bytes", 0) // 1024,
        "free_ram_kb": memory.get("free_ram_bytes", 0) // 1024,
        "available_ram_kb": memory.get("available_ram_bytes", 0) // 1024,
        "cached_kb": memory.get("cached_ram_bytes", 0) // 1024,
        "in_dim_0": input_shape[0],
        "in_dim_1": input_shape[2],
        "in_dim_2": input_shape[3],
        "in_dim_3": input_shape[1],
        "device_analytics": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "cpu_info": {
                "cpu_cores": os.cpu_count(),
                "processors": [{"vendor_id": platform.processor(), "model": platform.machine()}],
                "torch_threads": torch.get_num_threads(),
                "util_percent_snapshot_mean": float(statistics.mean(cpu_util)) if cpu_util else None,
            },
            "gpu_info": {
                "gpu_name": gpu_name,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "providers_available": ort.get_available_providers(),
                "util_percent_snapshot_mean": float(statistics.mean(gpu_util)) if gpu_util else None,
            },
            "timeline": timeline,
            "profile": {
                "peak_cpu_rss_bytes": max(process_rss) if process_rss else None,
                "peak_gpu_mb": max(gpu_memory) if gpu_memory else None,
                "torch_peak_gpu_allocated_mb": torch_peak_gpu_allocated_mb,
                "cpu_util_percent_samples": cpu_util,
                "gpu_util_percent_samples": gpu_util,
                "utilization_sampling": "before_eval, after_cpu, after_gpu snapshots",
                "input_source": input_source,
            },
        },
        "gpu_preflight": gpu_preflight,
        "gpu_preflight_snapshot": gpu_preflight_snapshot,
    }
    del cpu_model
    gc.collect()
    return profile


def _dataset_normalization(dataset):
    if dataset in {"cifar-10", "cifar-100"}:
        return [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    if dataset == "mnist":
        return [0.1307], [0.3081]
    if dataset == "imagenette":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [], []


def _inference_command():
    return [sys.executable, "-m", "ab.bot.port", "--ros-args"]


def _camera_emulator_command(args):
    return [
        sys.executable,
        "-m",
        "ab.bot.synthetic_camera",
        "--ros-args",
        "-p",
        f"image_topic:={args.input_topic}",
        "-p",
        f"publish_rate_hz:={args.camera_rate_hz}",
    ]


def _report_path(model_name, args):
    return args.results_dir / model_name / f"robot_{args.dataset}_{socket.gethostname()}.json"


def _has_complete_record(model_name, args):
    path = _report_path(model_name, args)
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        record.get("valid") is True
        and record.get("dataset") == args.dataset
        and record.get("profile_version") == 1
    )


def _merge_profile(model_name, args, profile):
    path = _report_path(model_name, args)
    record = json.loads(path.read_text(encoding="utf-8"))
    analytics = record.setdefault("device_analytics", {})
    ort_version = analytics.get("onnxruntime_version")
    ort_providers = analytics.get("providers")
    ort_timestamp = analytics.get("timestamp")
    ort_provider = ort_providers[0] if ort_providers else record.get("onnxruntime_provider")
    record["onnxruntime_latency_ns"] = record.get(
        "onnxruntime_latency_ns", record.get("latency_ns")
    )
    record["onnxruntime_provider"] = ort_provider
    record["runtime"] = "PyTorch and ONNX Runtime"
    for field in (
        "cpu_duration", "cpu_min_duration", "cpu_max_duration", "cpu_median_duration",
        "cpu_p95_duration", "cpu_std_dev",
        "cpu_runs", "cpu_error", "gpu_duration", "gpu_min_duration",
        "gpu_max_duration", "gpu_median_duration", "gpu_p95_duration", "gpu_std_dev",
        "gpu_runs", "gpu_error", "duration",
        "unit", "iterations", "warmup_runs", "torch_cpu_threads", "torch_gpu_name",
        "cpu_finite", "gpu_finite", "cpu_gpu_match", "cpu_gpu_max_abs_error",
        "cpu_gpu_top1_match",
        "cpu_gpu_rtol", "cpu_gpu_atol", "pytorch_onnx_match",
        "pytorch_onnx_max_abs_error", "gpu_onnx_match", "gpu_onnx_max_abs_error",
        "total_ram_kb", "free_ram_kb",
        "available_ram_kb", "cached_kb", "in_dim_0", "in_dim_1", "in_dim_2",
        "in_dim_3", "task", "metric", "training_accuracy", "checkpoint_repository",
        "checkpoint_epoch",
    ):
        record[field] = profile[field]
    record["gpu_backend"] = "PyTorch CUDA" if profile["gpu_runs"] else None
    record["cpu_backend"] = "PyTorch CPU"
    record["profile_version"] = 1
    analytics.update(profile["device_analytics"])
    analytics["onnxruntime_version"] = ort_version
    analytics["onnxruntime_providers"] = ort_providers
    analytics["onnxruntime_timestamp"] = ort_timestamp
    analytics["torch_version"] = profile["torch_version"]
    record["inference_time_scope"] = (
        "pytorch_cpu_forward_only" if profile["unit"] == "CPU" else "pytorch_cuda_forward_only"
    )
    record["cpu_runs_ns"] = profile["cpu_runs_ns"]
    record["gpu_runs_ns"] = profile["gpu_runs_ns"]
    record["onnxruntime_consistency_providers"] = profile[
        "onnxruntime_consistency_providers"
    ]
    record["gpu_preflight"] = {
        "snapshot": profile["gpu_preflight_snapshot"],
        "assessment": profile["gpu_preflight"],
    }

    cpu_correct = (
        profile["cpu_runs"] > 0
        and profile["cpu_finite"]
        and profile["pytorch_onnx_match"]
    )
    gpu_correct = (
        profile["gpu_runs"] > 0
        and profile["gpu_finite"] is True
        and profile["gpu_onnx_match"] is True
    )

    if cpu_correct or gpu_correct:
        valid_backends = {}
        if cpu_correct:
            valid_backends["CPU"] = profile["cpu_duration"]
        if gpu_correct:
            valid_backends["GPU"] = profile["gpu_duration"]
        fastest_backend = min(valid_backends, key=valid_backends.get)
        record["duration"] = valid_backends[fastest_backend]
        record["unit"] = (
            "CPU" if fastest_backend == "CPU" else profile["torch_gpu_name"]
        )
        record["inference_time_scope"] = (
            "pytorch_cpu_forward_only"
            if fastest_backend == "CPU"
            else "pytorch_cuda_forward_only"
        )

    validation_errors = []
    if not cpu_correct and not profile["cpu_finite"]:
        validation_errors.append("PyTorch CPU output contains non-finite values")
    elif not cpu_correct:
        validation_errors.append("PyTorch CPU and ONNX outputs do not match")
    if not gpu_correct and profile["gpu_runs"] < 1:
        validation_errors.append(profile["gpu_error"] or "PyTorch CUDA profiling did not run")
    elif not gpu_correct and profile["gpu_finite"] is not True:
        validation_errors.append("PyTorch CUDA output contains non-finite values")
    elif not gpu_correct:
        validation_errors.append("PyTorch CUDA and ONNX outputs do not match")
    if profile["cpu_gpu_match"] is False:
        validation_errors.append("PyTorch CPU and CUDA outputs do not match")
    if profile["cpu_gpu_top1_match"] is False:
        validation_errors.append("PyTorch CPU and CUDA top-1 predictions do not match")
    if record.get("failed_frames", 0):
        validation_errors.append(
            f'{record["failed_frames"]} image frame(s) failed during inference'
        )
    record["valid"] = bool(
        record.get("valid")
        and (cpu_correct or gpu_correct)
        and not record.get("failed_frames", 0)
    )
    if not record["valid"]:
        record["error_message"] = "; ".join(validation_errors) or "Inference validation failed"
    else:
        record["error_message"] = None

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return record["valid"]


def _run_model(model_path, model_name, args):
    report_path = _report_path(model_name, args)
    if _has_complete_record(model_name, args) and not args.force:
        print(f"[SKIP] {model_name}: valid result already exists")
        return True
    if args.force:
        report_path.unlink(missing_ok=True)

    mean, std = _dataset_normalization(args.dataset)
    if args.input_mean is not None:
        mean = args.input_mean
    if args.input_std is not None:
        std = args.input_std
    params = {
        "model_path": model_path.resolve(),
        "input_topic": args.input_topic,
        "output_topic": args.output_topic,
        "results_dir": args.results_dir.resolve(),
        "dataset": args.dataset,
        "synthetic_input": args.emulate_camera,
        "camera_rate_hz": args.camera_rate_hz if args.emulate_camera else 0.0,
        "warmup_runs": args.warmup_runs,
        "iterations": args.iterations,
        "input_scale": args.input_scale,
        "input_mean": mean,
        "input_std": std,
        "exit_after_benchmark": True,
    }
    command = _inference_command()
    for key, value in params.items():
        if isinstance(value, Path):
            value = str(value)
        if isinstance(value, list):
            value = "[" + ",".join(map(str, value)) + "]"
        command.extend(["-p", f"{key}:={value}"])

    print(f"[RUN] {model_name}: {' '.join(command)}", flush=True)
    camera_publisher = None
    try:
        if args.emulate_camera:
            camera_command = _camera_emulator_command(args)
            print(f"[CAMERA] {' '.join(camera_command)}", flush=True)
            camera_publisher = subprocess.Popen(camera_command)
        result = subprocess.run(command, timeout=args.model_timeout, check=False)
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {model_name}: benchmark exceeded {args.model_timeout}s")
        return False
    except OSError as error:
        print(f"[FAIL] {model_name}: could not start ROS process: {error}")
        return False
    finally:
        if camera_publisher is not None:
            camera_publisher.send_signal(signal.SIGINT)
            try:
                camera_publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                camera_publisher.kill()
                camera_publisher.wait()
    if result.returncode != 0 or not report_path.exists():
        print(f"[FAIL] {model_name}: ROS node exited with status {result.returncode}")
        return False
    return True


def parser():
    ap = argparse.ArgumentParser(
        description="Export NN-Dataset image classifiers and benchmark them with a ROS 2 image stream."
    )
    selection = ap.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", help="One NN-Dataset architecture name, e.g. AirNet")
    selection.add_argument("--all-models", action="store_true", help="Run every matching architecture/checkpoint")
    ap.add_argument("--dataset", default="cifar-10", choices=("cifar-10", "cifar-100", "mnist", "imagenette"))
    ap.add_argument("--limit", type=int, help="Limit an --all-models run (useful for a short trial)")
    ap.add_argument("--input-topic", default="/camera/image_raw")
    ap.add_argument("--output-topic", default="/nn_bot/output")
    ap.add_argument(
        "--emulate-camera",
        action="store_true",
        help="Publish synthetic 640x480 RGB frames instead of requiring a camera",
    )
    ap.add_argument("--camera-rate-hz", type=float, default=30.0)
    ap.add_argument(
        "--gpu-busy-policy",
        choices=("warn", "fail"),
        default="warn",
        help="Warn or skip profiling when a GPU exceeds the configured pre-run load limits",
    )
    ap.add_argument("--gpu-utilization-limit", type=float, default=20.0)
    ap.add_argument("--gpu-memory-limit-percent", type=float, default=20.0)
    ap.add_argument("--onnx-dir", type=Path, default=Path("_work/onnx"))
    ap.add_argument("--results-dir", type=Path, default=Path("stats"))
    ap.add_argument("--export-only", action="store_true", help="Export ONNX files without starting ROS inference")
    ap.add_argument("--warmup-runs", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--input-scale", type=float, default=1.0 / 255.0)
    ap.add_argument("--input-mean", type=float, nargs="+")
    ap.add_argument("--input-std", type=float, nargs="+")
    ap.add_argument("--model-timeout", type=float, default=1800)
    ap.add_argument("--force", action="store_true", help="Re-export and benchmark existing results")
    return ap


def main(args=None):
    ap = parser()
    args = ap.parse_args(args)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")
    if args.iterations < 1 or args.warmup_runs < 0 or args.model_timeout <= 0:
        ap.error("iterations and model-timeout must be positive; warmup-runs cannot be negative")
    if not math.isfinite(args.camera_rate_hz) or args.camera_rate_hz <= 0:
        ap.error("--camera-rate-hz must be finite and positive")
    if (
        not math.isfinite(args.gpu_utilization_limit)
        or not 0 <= args.gpu_utilization_limit <= 100
        or not math.isfinite(args.gpu_memory_limit_percent)
        or not 0 <= args.gpu_memory_limit_percent <= 100
    ):
        ap.error("GPU utilization and memory limits must be finite percentages from 0 to 100")
    if (args.input_mean is None) != (args.input_std is None):
        ap.error("--input-mean and --input-std must be given together")

    try:
        models, hf_hub_download = discover_models(args.dataset, args.model)
    except (ImportError, RuntimeError, ValueError) as error:
        ap.error(str(error))
    if args.limit is not None:
        models = models[:args.limit]

    print(f"[SETUP] {len(models)} model(s), dataset={args.dataset}")
    completed = 0
    failed = []
    for index, model in enumerate(models, 1):
        name = model["nn"]
        print(f"\n[{index}/{len(models)}] {name}", flush=True)
        model_path = args.onnx_dir / args.dataset / f"{name}.onnx"
        if args.force:
            model_path.unlink(missing_ok=True)
        try:
            if not model_path.exists():
                export_model(model, model_path, hf_hub_download)
            if args.export_only:
                completed += 1
                print(f"[EXPORTED] {model_path}")
                continue
            if _has_complete_record(name, args) and not args.force:
                print(f"[SKIP] {name}: valid CPU/GPU profile already exists")
                completed += 1
                continue
            profile = profile_model(
                model,
                model_path,
                hf_hub_download,
                iterations=args.iterations,
                warmup_runs=args.warmup_runs,
                gpu_utilization_limit=args.gpu_utilization_limit,
                gpu_memory_limit_percent=args.gpu_memory_limit_percent,
                gpu_busy_policy=args.gpu_busy_policy,
            )
            for warning in profile["gpu_preflight"]["warnings"]:
                print(f"[WARN] {name}: {warning}", flush=True)
            if profile["gpu_preflight"]["busy"] and args.gpu_busy_policy == "fail":
                print(f"[FAIL] {name}: GPU became busy before CUDA profiling", flush=True)
                failed.append(name)
                continue
            if _run_model(model_path, name, args):
                if _merge_profile(name, args, profile):
                    completed += 1
                else:
                    failed.append(name)
            else:
                failed.append(name)
        except Exception as error:
            print(f"[FAIL] {name}: {error}", flush=True)
            failed.append(name)

    print(f"\n[SUMMARY] successful={completed}, failed={len(failed)}")
    if failed:
        print("Failed models: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
