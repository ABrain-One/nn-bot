# NN-Bot

ROS 2 inference and benchmarking for neural networks running on robotic systems.
The initial runtime accepts ONNX image models, subscribes to a ROS image topic,
publishes model outputs, and writes a per-model latency/device record after the
configured number of measured frames. It follows NN-VR's portable ONNX handoff
and NN-Lite's measured-run statistics and device records.

## Repository Layout

```text
nn-bot/
├── ab/
│   ├── bot/
│   │   ├── dataset_pipeline.py  # Export, validate, and benchmark NN-Dataset models
│   │   ├── port.py              # ROS 2 ONNX inference node
│   │   ├── runtime.py           # Preprocessing, telemetry, and timing helpers
│   │   └── synthetic_camera.py  # Synthetic image stream support
│   └── nn/transform/            # Preprocessing transforms and manifests
├── resource/                    # ROS package index marker
├── scripts/                     # Helper scripts, including background runs
├── stats/                       # Saved benchmark JSON and NN-Stat CSV records
│   └── <ModelName>/provider-comparison/
│       ├── cpu/                 # CPU-only ONNX Runtime benchmark
│       ├── cuda/                # CUDA-only ONNX Runtime benchmark
│       └── cuda-diagnostic/     # CUDA node-placement profile and summary
├── tests/                       # Runtime and dataset-pipeline tests
├── package.xml                  # ROS package dependencies and metadata
├── requirements*.txt            # Python dependency sets
├── setup.py / setup.cfg         # Python and ament package configuration
└── README.md                    # Setup, usage, and benchmark guidance
```

Generated models, preprocessing manifests, and temporary benchmark outputs are
kept under the ignored `_work/` directory unless selected results are copied
into `stats/`.

## Requirements

- Ubuntu 22.04 with ROS 2 Humble and `colcon`
- Python 3.10+
- An exported ONNX model with one rank-4, batch-one image input
- `cv_bridge`, `sensor_msgs`, and `std_msgs`
- Python packages from `requirements.txt`

Install ROS 2 Humble with `bash install-ros.sh`, then install the Python
dependencies in an environment that can see ROS's apt-installed Python modules:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a supported NVIDIA CUDA environment, replace the CPU ONNX Runtime wheel
with the GPU build so the ROS node and ONNX consistency check can use
`CUDAExecutionProvider`:

```bash
python -m pip uninstall -y onnxruntime
python -m pip install -r requirements-gpu.txt
```

The pipeline requests CUDA first and falls back to CPU if CUDA is unavailable.
Confirm that CUDA is listed by `onnxruntime.get_available_providers()` before
interpreting ONNX timings as GPU measurements. Jetson and other ARM devices may
need the ONNX Runtime wheel supplied for their JetPack/CUDA release instead of
the standard PyPI package.

The ROS package exposes runtime choices as extras for installs that use pip:
`pip install '.[cpu]'` or `pip install '.[gpu]'`. The regular requirements files
remain an alternative for environment-based installs.

For the NN-Dataset export pipeline, install its additional dependencies. On a
CPU-only machine, reinstall the CPU PyTorch wheels after them so dependency
resolution does not leave a CUDA build in the environment:

```bash
python -m pip install -r requirements-dataset.txt
python -m pip install --force-reinstall torch torchvision \
	--index-url https://download.pytorch.org/whl/cpu
```

The runner imports model architecture files from an NN-Dataset checkout and
downloads epoch-50 checkpoints and metadata from Hugging Face. From the
repository root, clone NN-Dataset alongside this repository and put both roots
on `PYTHONPATH`, with nn-bot first so its `ab` namespace can also find `ab.nn`:

```bash
git clone --depth 1 https://github.com/ABrain-One/nn-dataset.git ../nn-dataset
export PYTHONPATH="$PWD:$PWD/../nn-dataset${PYTHONPATH:+:$PYTHONPATH}"
```

## Build

From the repository root:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
```

## Run One ONNX Model

Point the node at an ONNX model and the robot's image topic:

```bash
python -m ab.bot.port --ros-args \
	-p model_path:=/path/to/model.onnx \
	-p input_topic:=/camera/image_raw \
	-p output_topic:=/nn_bot/output
```

The output is a `std_msgs/Float32MultiArray` containing the first model output.
Ten incoming frames are used for warm-up by default; the next 500 frames are timed
(configurable with `iterations`, recommended 500–1000 frames).
Set `warmup_runs`, `iterations`, `deadline_ms`, `cpu_affinity`, `onnx_num_threads`,
`input_scale`, `input_mean`, and `input_std` as ROS parameters for models without
a preprocessing manifest. If `<model>.preprocess.json` sits beside the ONNX file,
the node loads it automatically; use `preprocessing_manifest` to specify another
path. A manifest takes precedence over the scalar preprocessing parameters.
Without a manifest, the legacy preprocessing resizes with nearest-neighbor
sampling, converts RGB pixels to float32 in `[0, 1]`, and supports NCHW or NHWC
input layouts.

The published output is the raw first model output by default. Set
`-p output_normalization:=l2` only when downstream consumers require L2-normalized
values. Benchmark records identify the selected output normalization and the
preprocessing manifest or legacy parameters used.

Results are written to
`stats/<model>/robot_<dataset>_<hostname>.json` by default. Use
`--onnx-provider cpu` or `--onnx-provider cuda` to run a strict ONNX Runtime
CPU-only or CUDA-only session. CUDA mode disables CPU execution-provider
fallback; a CPU run selects only the CPU provider. A CUDA run fails rather than
silently using CPU if CUDA is unavailable or cannot execute the model. The
default `auto` mode keeps CUDA-then-CPU provider behavior. ONNX Runtime may still
list its CPU provider in `onnxruntime_active_providers` as a registered provider
in strict CUDA mode; the mode disables CPU fallback for graph execution.

Run CPU and CUDA comparisons into separate result directories so one does not
overwrite the other:

```bash
python -m ab.bot.dataset_pipeline --model AirNet --dataset cifar-10 \
	--emulate-camera --onnx-provider cpu --results-dir _work/provider-runs/cpu

python -m ab.bot.dataset_pipeline --model AirNet --dataset cifar-10 \
	--emulate-camera --onnx-provider cuda --results-dir _work/provider-runs/cuda
```

In the JSON, `onnxruntime_execution_mode`, `onnxruntime_requested_providers`,
`onnxruntime_active_providers`, and `onnxruntime_provider` describe the ROS
node's ONNX Runtime session. In strict CUDA mode `onnxruntime_provider` is
CUDA even if the active-provider list also includes CPU; `onnxruntime_profile`
is the per-node evidence for diagnostic runs. `onnxruntime_latency_ns` measures
that selected session's `session.run()` time. The separate `pytorch_cpu_*` and
`pytorch_cuda_*` fields describe PyTorch correctness/validation runs; legacy
`cpu_runs` and `gpu_runs` are aliases for those PyTorch runs and are not counts
of ROS ONNX Runtime executions.

Use `--enable-onnx-profiling` in a separate diagnostic run to inspect actual
per-node execution-provider events. Profiling adds overhead, so its latency
samples are diagnostic only. With `auto`, a trace may show nodes split between
CUDA and CPU; strict `cpu`/`cuda` modes should show only the selected provider.

The standalone ONNX node's primary `latency_ns` measures image conversion and
preprocessing through ONNX inference and publishing that output, including any
optional output normalization. `onnxruntime_latency_ns` separately measures only
the `session.run()` call and is used for NN-Stat exports. `callback_latency_ns`
includes callback-entry-to-publish overhead. Every timing summary includes `avg`, `median`, `p95`,
`p99`, `min`, `max`, `std`, and individual `samples`. Standard deviation uses
population standard deviation (dividing by N, `ddof=0`), matching nn-lite's
convention. `deadline_missed_frames` reports how many frames exceeded the
configured deadline (default: 33 ms at 30 Hz).

Records keep measurement and inference validation separate:
`run_complete` means the requested number of frames was collected;
`measurement_conditions_valid` reports clock, camera-rate, dropped-frame, CPU,
and GPU-preflight checks; `preprocessing_provenance_valid` requires a model
preprocessing manifest; and `inference_correctness_valid` is set by the
PyTorch/ONNX comparison. `performance_measurement_valid` is false when conditions
are unsuitable, preprocessing is unverified, or ONNX profiling is enabled.
`valid` is true only when the run is complete, the performance measurement is
comparable, and at least one tested backend passes correctness checks.

To ensure controlled measurement conditions:
- **CPU Pinning**: Pin the process to a specific core using `cpu_affinity` (e.g. `-p cpu_affinity:=0`).
- **CPU telemetry**: Governor and temperature checks are retained. Per-core
  frequency and utilization are sampled during timed image processing and
  reported under `device_analytics.cpu_telemetry`; a single system-average
  before/after clock snapshot is descriptive only and does not invalidate a run.
- **ONNX Thread Limit**: ONNX Runtime intra-op threads are constrained via `onnx_num_threads` (default: 1) to avoid unconstrained thread contention.
- **Temperature Logging**: System and GPU temperatures are sampled and logged in `device_analytics.temperatures`.
- **Event-Driven Execution**: Node execution uses blocking `spin()` instead of polling `spin_once(0.1)` in a tight loop.

The `input_shape` field retains the model's actual tensor layout; `in_dim_0`
through `in_dim_3` use NN-Lite's batch, height, width, channels order so records
can be merged with that schema. NPU fields (`npu_duration`, `npu_min_duration`,
`npu_max_duration`, `npu_std_dev`, `npu_runs`, `npu_error`) are included in the
schema for compatibility. `unit` identifies the host hardware.
`onnxruntime_active_providers` lists the active ONNX Runtime session providers.
`onnxruntime_provider` identifies the selected provider in strict CPU/CUDA mode,
or a single-provider auto session. In mixed-provider auto mode, per-node
execution is unverified unless profiling is enabled. A requested or active
CUDA provider alone does not prove that CUDA executed every model node.

For a separate diagnostic run, pass `--enable-onnx-profiling` to save an
ONNX Runtime trace and summarize per-node event counts, unique nodes, and
operators by execution provider in `onnxruntime_profile`. Profiling adds
overhead; those latency samples are therefore explicitly marked diagnostic and
must not be used as performance results. Keep profiling disabled for timing runs.
For example:

```bash
python -m ab.bot.dataset_pipeline --model MobileNetV3 \
	--dataset cifar-10 --emulate-camera --enable-onnx-profiling \
	--results-dir _work/provider-diagnostics
```

For NN-Stat interoperability, each merged result also includes an `nn_stat`
object in the JSON record and writes
`stats/<model>/robot_<dataset>_<hostname>_nn_stat.csv`. The CSV contains the
`task`, `dataset`, `epoch`, `metric`, `accuracy`, `duration`, and `nn` columns
used by NN-Stat's `process_data()`; it also includes source and unit columns.
`metric` uses NN-Stat's `acc` code. `accuracy` is the checkpoint's reported
training accuracy, not accuracy re-measured by this inference run. `duration`
is the mean `session.run()` time in seconds per image, not training duration.
The JSON `duration` remains the full image-preprocessing-through-publish latency.

The CSV can be passed directly to NN-Stat's processor:

```python
import pandas as pd
from ab.stat.util.process_data import process_data

aggregated = process_data(pd.read_csv(
    "stats/AirNet/robot_cifar-10_<hostname>_nn_stat.csv"
))
```

## Run NN-Dataset Models

Build and source the ROS package in the same shell:

```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -m colcon build --symlink-install
source install/setup.bash
```

Run one NN-Dataset image-classification model by architecture name:

```bash
python -m ab.bot.dataset_pipeline --model MobileNetV3 \
	--dataset cifar-10 \
	--input-topic /camera/image_raw
```

MobileNetV3 is a compact checkpoint-backed example (591,802 parameters in the
NN-Dataset configuration) and has been exported and run through the ROS image
callback in this workspace.

For each model, the pipeline loads one transformed NN-Dataset training sample,
runs 10 PyTorch warm-ups and 20 timed inferences on CPU, then repeats on CUDA
when PyTorch reports an available CUDA device. CPU time uses a monotonic host
clock; CUDA time uses events with synchronization after each run. It also runs
the same input through the exported ONNX model and compares outputs. CPU/CUDA
logits are checked with `rtol=0.01, atol=0.002`; PyTorch/ONNX outputs use
`rtol=0.001, atol=0.0001`. The PyTorch/ONNX correctness check uses the CPU ONNX
provider as a stable reference; the ROS ONNX measurement prefers CUDA when
available and records its active provider. Finite-value and top-class agreement
are recorded. Every timing summary keeps its individual samples and includes
mean, median, p95, p99, and max.

The pipeline writes a `<model>.preprocess.json` sidecar beside each ONNX export
and passes it to the ROS node. Supported deterministic torchvision operations
are translated to the manifest; random horizontal-flip augmentation is omitted
for inference. Unsupported transforms fail with an explicit error rather than
silently using different preprocessing. `--input-scale`, `--input-mean`, and
`--input-std` customize the generated manifest. For custom ONNX models, provide
an equivalent sidecar or use the legacy ROS preprocessing parameters. A
manifest's `operations` run in order; supported operations are `resize`
(`shorter` or `exact`), `center_crop`, `grayscale`, `to_tensor`, and `normalize`.
For example:

```json
{
  "version": 1,
  "operations": [
    {"type": "resize", "size": 128, "mode": "shorter", "interpolation": "bilinear"},
    {"type": "center_crop", "size": 128},
    {"type": "to_tensor", "scale": 0.00392156862745098},
    {"type": "normalize", "mean": [0.4914, 0.4822, 0.4465], "std": [0.247, 0.2435, 0.2616]}
  ]
}
```

Model outputs are published raw by default. Use `--output-normalization l2` to
retain L2 normalization for consumers that depend on the previous behavior.

Before profiling each model, the pipeline samples NVIDIA GPU utilization and
memory use. It warns by default when utilization or memory use exceeds 20%;
set `--gpu-utilization-limit` and `--gpu-memory-limit-percent` to adjust those
limits. Use `--gpu-busy-policy fail` to skip the profile and mark the model as
failed when a measured threshold is exceeded. When `nvidia-smi` is unavailable,
the record notes that GPU load could not be checked.

The ROS node measures 500 image-message inferences after 10 warm-ups by default
(configurable with `--iterations`, recommended 500–1000 frames). Its primary
`latency_ns` interval starts before image conversion/preprocessing and ends after
publishing the L2-normalized model output, reporting `p99`, `max`, and
`deadline_missed_frames` against the configured deadline (default: 33 ms at 30 Hz).
For a reproducible ROS image-stream emulation,
add `--emulate-camera`; the pipeline starts the native `image_tools/cam2image`
publisher in burger-pattern mode to send stamped 640x480 RGB test frames at
30 Hz by default. Set
`--camera-rate-hz` to change the rate. For example:

```bash
python -m ab.bot.dataset_pipeline --model AirNet \
	--dataset cifar-10 \
	--emulate-camera \
	--camera-rate-hz 30
```

Use the pipeline-generated preprocessing manifest for measurements. If the ROS
node runs without one, it still accepts images using legacy parameters, but
marks preprocessing provenance unverified and the performance result invalid.
The node checks and records CPU telemetry before and after the ROS run and takes
a GPU utilization/memory preflight immediately before constructing the ONNX
session. Thresholds can be set with `--gpu-utilization-limit` and
`--gpu-memory-limit-percent`; the node reports both the ROS-session and
PyTorch-profile preflights separately.

The inference node records separate `end_to_end_latency_ns` (image header stamp
to output publish) and `callback_latency_ns` (callback entry to output publish)
metrics; their samples are never combined. `source_to_callback_latency_ns`
separates camera/header-to-callback delay from callback processing, so an
end-to-end-minus-callback difference should not automatically be attributed to
ROS callback queuing. Frames without a usable header stamp
are omitted from `end_to_end_latency_ns`. If a header timestamp is later than
the ROS callback-entry or publish time, the frame is counted in
`clock_inconsistency_frames`, excluded from the header-based metric, and makes
the run invalid rather than substituting callback timing. The `emulator` field
means Android emulator and is `false` for this ROS-hosted node. Synthetic runs
separately record `synthetic_input: true` and the requested camera rate;
physical camera runs record `synthetic_input: false`. These figures measure
the host and ROS path under the configured synthetic stream, not camera optics,
sensor acquisition, or the physical robot's environment.

The native synthetic publisher is used to avoid the Python serialization
throughput limit for 640x480 ROS image messages. It uses the ROS image-tools
publisher with reliable, keep-last QoS (depth 5), matched by the inference
subscriber for synthetic streams, and remaps its `/image` topic to the
configured input topic. Physical-camera subscriptions retain best-effort QoS.
Its
`dropped_frames` count estimates missed
synthetic camera frames from gaps between received source timestamps; it is
`null` for physical-camera runs, where the source frame rate is not configured.
Synthetic runs allow up to 0.5% estimated timestamp-derived frame loss, provided
the requested input-rate target is still met; larger losses invalidate the run.
The estimated loss fraction and tolerance are recorded with the raw drop count.
Runs whose GPU preflight
exceeds configured utilization or memory limits. CPU timing runs are invalidated
when the captured governor is `powersave` or a sampled temperature is at least
85 C. Per-core frequency and utilization samples are collected during the
measured window for diagnosis, rather than invalidating runs based on idle-core
averages taken before or after inference. Invalidating conditions are recorded
in `error_message`; these checks do not imply CUDA execution.

### Set the CPU frequency governor (Linux)

The governor is a system-wide CPU policy setting; the benchmark does not change
it automatically. Check which governors the CPU driver supports and record the
current policy before changing it:

```bash
cpupower frequency-info --governors
cpupower frequency-info --policy
```

Set the governor for the benchmark with administrator privileges:

```bash
sudo cpupower frequency-set --governor performance
```

To switch back to the energy-saving governor afterward:

```bash
sudo cpupower frequency-set --governor powersave
```

Verify the setting on every CPU frequency policy:

```bash
grep -H . /sys/devices/system/cpu/cpufreq/policy*/scaling_governor
```

Use only a governor listed by `cpupower frequency-info --governors`. If
`cpupower` is not installed, install the `cpupower` or `linux-tools` package
provided by your Linux distribution. `performance` requests a high-performance
policy and `powersave` requests an energy-saving policy; neither fixes the CPU
at a specific clock, and hardware, thermal, or power limits can still affect
frequency. Restore the original governor after benchmarking.

To find a sustainable camera rate, run separate benchmarks at (for example)
15, 20, 25, and 30 Hz, using a distinct `--results-dir` for each run. Compare
`input_rate_target_met`, `observed_input_rate_hz`, and `dropped_frames` before
comparing latency. A lower camera-rate run characterizes that rate; it does not
demonstrate that the pipeline sustains 30 Hz. Keep model, preprocessing
manifest, warmups, iteration count, CPU settings, and GPU preflight limits fixed
across the sweep. Do not change system-wide CPU governors automatically; set
and document the environment intentionally before a controlled measurement.
For example, use `_work/rate-sweep/15hz`, `_work/rate-sweep/20hz`, and so on as
the four result directories; repeat each rate into a separate repeat-specific
directory and do not overwrite or combine raw records.

For real-robot runs, omit `--emulate-camera` and point `--input-topic` at the
physical camera. The record uses `synthetic_input: false`; the
message-stamp-to-output metric then includes the camera driver and ROS delivery
delays to the inference callback. Run navigation/control nodes as usual during
the measurement when you want to capture their shared-system impact. Preserve
the GPU preflight warnings and load snapshot when comparing runs.

CPU/GPU utilization and system/GPU memory are snapshots
before and after each backend, not continuous averages; the record labels them
accordingly. If PyTorch CUDA is unavailable or fails, the CPU results are kept
and the GPU error is recorded rather than represented as a zero latency.

The first non-export-only run loads a transformed sample through NN-Dataset;
for CIFAR-10 this can download its roughly 170 MB training archive. Later runs
reuse the local dataset cache.

To export a different model without starting a camera benchmark, use
`--export-only`. For example, export AirNet for CIFAR-10:

```bash
python -m ab.bot.dataset_pipeline --model AirNet \
	--dataset cifar-10 \
	--export-only
```

The ONNX file is written to
`_work/onnx/cifar-10/AirNet.onnx`. Deploy it with the regular ROS node:

```bash
python -m ab.bot.port --ros-args \
	-p model_path:=$PWD/_work/onnx/cifar-10/AirNet.onnx \
	-p input_topic:=/camera/image_raw \
	-p output_topic:=/nn_bot/output \
	-p dataset:=cifar-10
```

The node publishes the first model output on `/nn_bot/output` and records stats
after 20 measured frames by default. It then continues running for deployment;
stop it with Ctrl-C. The dataset pipeline's normal `--model ...` command instead
exports the model, runs the CPU/CUDA and live ROS/ONNX measurements, merges the
statistics and correctness checks into one record, then exits automatically.

Run all matching NN-Dataset architecture modules and epoch-50 checkpoints for
the chosen dataset:

```bash
python -m ab.bot.dataset_pipeline --all-models \
	--dataset cifar-10 \
	--input-topic /camera/image_raw
```

An all-model run can take a long time. Use a small limit first to check the
camera topic, dependencies, and model preprocessing:

```bash
python -m ab.bot.dataset_pipeline --all-models --dataset cifar-10 --limit 3
```

### Run in the Background

Start a detached run from the repository root. The launcher saves a PID and a
log under `_work/background/`:

```bash
NN_DATASET_ROOT="$PWD/../nn-dataset" bash scripts/run-background.sh \
	--model AirNet --dataset cifar-10 --input-topic /camera/image_raw
```

For every matching model, use `--all-models` instead of `--model AirNet`. Watch
progress with `tail -f _work/background/<run-id>.log`; stop the job with
`kill "$(cat _work/background/<run-id>.pid)"`.

The launcher tries to inhibit systemd lid-switch and sleep handling. If that
permission is unavailable, it warns and still detaches the job, but the run
will pause if the computer suspends. Add `--require-lid-inhibitor` before the
pipeline arguments to refuse to start unless the inhibitor can be acquired.
Even with an inhibitor, keep the laptop powered and ventilated during long
profiles.

The runner exports ONNX models under `_work/onnx/<dataset>/`, then starts the
ROS inference node for each model. It listens to the live image topic, warms up
the model, measures the configured number of frames, and writes a record to
`stats/<model>/robot_<dataset>_<hostname>.json` before advancing. Failed image
frames are counted and make the record invalid. Overall validity requires a
complete ROS run, valid measurement conditions and preprocessing provenance,
and at least one PyTorch backend passing the ONNX consistency check; CUDA is
optional when unavailable or unsupported. The record reports each component
separately so a correct inference can be distinguished from an unsuitable
performance measurement. Runs are invalidated for dropped or under-rate
synthetic input, a busy GPU preflight, or CPU telemetry indicating a powersave
governor, a clock below 60% of maximum, or a temperature of at least 85 C.
`duration`, `unit`, and `inference_time_scope` in the merged record retain the
ROS image-to-published-output measurement. The validated PyTorch forward-pass
summary is kept separately in `pytorch_duration`, `pytorch_unit`, and
`pytorch_inference_time_scope`, alongside the per-backend CPU/CUDA metrics.
The `latency_ns` distribution captures detailed latency percentiles (including
`p95`, `p99`, and `max`) and deadline misses.
Dataset-specific filenames keep results for different datasets separate.
Existing valid
records for the selected dataset are skipped when resuming; pass `--force` to
export and benchmark again.
The default normalization is dataset-specific for CIFAR-10, CIFAR-100, MNIST,
and Imagenette. Use `--input-scale`, `--input-mean`, and `--input-std` to override
it when required by the model.

## Tests

The preprocessing and timing tests need only NumPy and run without ROS 2 or a
robot:

```bash
python3 -m unittest discover -s tests
```
