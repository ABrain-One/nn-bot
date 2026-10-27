# NN-Bot

ROS 2 inference and benchmarking for neural networks running on robotic systems.
The initial runtime accepts ONNX image models, subscribes to a ROS image topic,
publishes model outputs, and writes a per-model latency/device record after the
configured number of measured frames. It follows NN-VR's portable ONNX handoff
and NN-Lite's measured-run statistics and device records.

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
Ten incoming frames are used for warm-up by default; the next 20 are timed.
Set `warmup_runs`, `iterations`, `input_scale`, `input_mean`, and `input_std`
as ROS parameters to match the model's preprocessing. The default preprocessing
resizes with nearest-neighbor sampling, converts RGB pixels to float32 in
`[0, 1]`, and supports NCHW or NHWC input layouts.

Results are written to
`stats/<model>/robot_<dataset>_<hostname>.json` by default. Select an execution
provider with the `providers` parameter, for example
`-p providers:="[CUDAExecutionProvider, CPUExecutionProvider]"` when supported
by the installed ONNX Runtime build.

The standalone ONNX node's primary `latency_ns` measures image conversion and
preprocessing through ONNX inference, L2-normalization of its output, and
publishing that output.
`onnxruntime_latency_ns` separately reports only the selected ONNX Runtime
provider's `session.run()` latency. Use the NN-Dataset pipeline below for the
PyTorch CPU/CUDA profile and output-consistency checks.
The `input_shape` field retains the model's actual tensor layout; `in_dim_0`
through `in_dim_3` use NN-Lite's batch, height, width, channels order so records
can be merged with that schema. `unit` identifies the host hardware, while
`onnxruntime_provider` identifies the ONNX Runtime provider used for inference.

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
mean, median, and p95.

Before profiling each model, the pipeline samples NVIDIA GPU utilization and
memory use. It warns by default when utilization or memory use exceeds 20%;
set `--gpu-utilization-limit` and `--gpu-memory-limit-percent` to adjust those
limits. Use `--gpu-busy-policy fail` to skip the profile and mark the model as
failed when a measured threshold is exceeded. When `nvidia-smi` is unavailable,
the record notes that GPU load could not be checked.

The ROS node measures 20 image-message inferences after 10 warm-ups. Its primary
`latency_ns` interval starts before image conversion/preprocessing and ends after
publishing the L2-normalized model output; `onnxruntime_latency_ns` records the
nested `session.run()` interval alone. For a reproducible ROS image-stream emulation,
add `--emulate-camera`; the pipeline starts a separate ROS 2 publisher that
sends stamped 640x480 RGB test frames at 30 Hz by default. Set
`--camera-rate-hz` to change the rate. For example:

```bash
python -m ab.bot.dataset_pipeline --model AirNet \
	--dataset cifar-10 \
	--emulate-camera \
	--camera-rate-hz 30
```

The inference node records `end_to_end_latency_ns` from the image header stamp
through publishing the model output, covering input transport/queueing and the
full callback; it also records `callback_latency_ns` from callback entry through
output publishing. When a message has no usable timestamp or the ROS clocks are
inconsistent, the end-to-end metric falls back to callback timing, with counts
of each source recorded. The `emulator` field means Android emulator and is
`false` for this ROS-hosted node. Synthetic runs separately record
`synthetic_input: true` and the requested camera rate; physical camera runs
record `synthetic_input: false`. These figures measure the host and ROS path
under the configured synthetic stream, not camera optics, sensor acquisition,
or the physical robot's environment.

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
frames are counted and make the record invalid. The pipeline record is also
valid when at least one PyTorch backend completes and its output passes the
ONNX consistency check; CUDA is optional when unavailable or unsupported.
`duration`, `unit`, and `inference_time_scope` in the merged record retain the
ROS image-to-published-output measurement. The validated PyTorch forward-pass
summary is kept separately in `pytorch_duration`, `pytorch_unit`, and
`pytorch_inference_time_scope`, alongside the per-backend CPU/CUDA metrics. The
`onnxruntime_latency_ns` field retains the ROS node's ONNX `session.run()` timing.
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
