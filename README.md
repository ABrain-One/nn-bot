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
`stats/<model>/robot_<hostname>.json` by default. Select an execution
provider with the `providers` parameter, for example
`-p providers:="[CUDAExecutionProvider, CPUExecutionProvider]"` when supported
by the installed ONNX Runtime build.

This standalone ONNX node reports only the selected ONNX Runtime provider's
`session.run()` latency. Use the NN-Dataset pipeline below for the comparable
PyTorch CPU/CUDA profile and output-consistency checks.

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
`rtol=0.001, atol=0.0001`. Finite-value and top-class agreement are recorded.

The ROS node separately measures 20 live-message ONNX Runtime `session.run()`
calls after 10 warm-ups. Preprocessing, ROS transport, and publishing are outside
that latency interval. CPU/GPU utilization and system/GPU memory are snapshots
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
`stats/<model>/robot_<hostname>.json` before advancing. Each record includes its
dataset. Since this layout is flat, running the same model on the same host with
a different dataset replaces that model's previous record. Existing valid
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
