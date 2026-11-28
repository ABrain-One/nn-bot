"""Fold inference-only Conv–BatchNormalization pairs and compare CPU latency."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, helper, numpy_helper


def fold_conv_batchnorm(model):
    """Rewrite eligible top-level pairs in place and return the number folded."""
    graph = model.graph
    if any(entry.domain in ("", "ai.onnx") and entry.version < 9 for entry in model.opset_import):
        return 0
    if model.training_info or graph.quantization_annotation or graph.sparse_initializer:
        return 0
    if any(a.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS)
           for node in graph.node for a in node.attribute):
        return 0  # A nested graph can capture intermediate tensors by name.

    inputs = {value.name for value in graph.input}
    outputs = {value.name for value in graph.output}
    constants = {tensor.name: tensor for tensor in graph.initializer if tensor.name not in inputs}
    producers = {name: node for node in graph.node for name in node.output}
    consumers = Counter(name for node in graph.node for name in node.input)
    names = set(producers) | inputs | outputs | {tensor.name for tensor in graph.initializer}

    # The legacy exporter can alias identical BatchNorm parameters with Identity.
    for node in graph.node:
        if node.domain in ("", "ai.onnx") and node.op_type == "Identity" and node.input[0] in constants:
            constants[node.output[0]] = constants[node.input[0]]

    removed = set()
    for norm in graph.node:
        if norm.domain not in ("", "ai.onnx") or norm.op_type != "BatchNormalization":
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in norm.attribute}
        if (
            len(norm.input) != 5 or any(norm.output[1:]) or attributes.get("training_mode", 0)
            or not attributes.get("is_test", 1) or not attributes.get("spatial", 1)
        ):
            continue
        intermediate = norm.input[0]
        conv = producers.get(intermediate)
        if (conv is None or conv.op_type != "Conv" or conv.domain not in ("", "ai.onnx")
                or intermediate in outputs or consumers[intermediate] != 1):
            continue
        weight_name = conv.input[1]
        bias_name = conv.input[2] if len(conv.input) > 2 else ""
        parameter_names = [weight_name, *norm.input[1:]]
        if bias_name:
            parameter_names.append(bias_name)
        if any(name not in constants for name in parameter_names):
            continue
        weights = numpy_helper.to_array(constants[weight_name])
        scale, offset, mean, variance = [
            numpy_helper.to_array(constants[name]) for name in norm.input[1:]
        ]
        bias = numpy_helper.to_array(constants[bias_name]) if bias_name else np.zeros_like(mean)
        if weights.dtype not in (np.float32, np.float64):
            continue
        if any(value.shape != (weights.shape[0],) for value in (bias, scale, offset, mean, variance)):
            continue

        # BN(Conv(x)) = scale / sqrt(variance + epsilon) * (Conv(x) - mean) + offset.
        multiplier = scale / np.sqrt(variance + attributes.get("epsilon", 1e-5))
        fused_weights = weights * multiplier.reshape((-1,) + (1,) * (weights.ndim - 1))
        fused_bias = (bias - mean) * multiplier + offset
        if not np.isfinite(fused_weights).all() or not np.isfinite(fused_bias).all():
            continue

        del conv.input[1:]
        for kind, array in (("weight", fused_weights), ("bias", fused_bias)):
            name = f"{norm.output[0]}_folded_{kind}"
            while name in names:
                name += "_"
            names.add(name)
            graph.initializer.append(numpy_helper.from_array(array.astype(weights.dtype), name))
            conv.input.append(name)
        conv.output[0] = norm.output[0]
        removed.add(norm.output[0])

    if removed:
        nodes = [node for node in graph.node if node.output[0] not in removed
                 or node.op_type != "BatchNormalization"]
        live = outputs.copy()
        kept = []
        for node in reversed(nodes):
            if node.op_type == "Identity" and node.output[0] in constants and node.output[0] not in live:
                continue
            kept.append(node)
            live.update(node.input)
        initializers = [tensor for tensor in graph.initializer if tensor.name in live | inputs]
        del graph.node[:]
        graph.node.extend(reversed(kept))
        del graph.initializer[:]
        graph.initializer.extend(initializers)
    return len(removed)


def _benchmark_feeds(session):
    dtypes = {"tensor(float)": np.float32, "tensor(double)": np.float64}
    inputs = session.get_inputs()
    if any(item.type not in dtypes for item in inputs):
        raise ValueError("Benchmark requires floating-point tensor inputs")

    rng = np.random.default_rng(7)
    feeds = []
    for _ in range(8):
        feed = {}
        for item in inputs:
            shape = [dimension if isinstance(dimension, int) else 1 for dimension in item.shape]
            feed[item.name] = rng.standard_normal(shape).astype(dtypes[item.type])
        feeds.append(feed)
    return feeds


def _measure_session(session, feeds, reference, samples):
    # These validation calls also warm up the session before each timed repeat.
    for feed, expected in zip(feeds, reference):
        outputs = session.run(None, feed)
        for actual, wanted in zip(outputs, expected, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=1e-4, atol=1e-5)

    timings = []
    for sample in range(samples):
        feed = feeds[sample % len(feeds)]
        start = perf_counter_ns()
        session.run(None, feed)
        timings.append(perf_counter_ns() - start)
    return {
        "median_ms": median(timings) / 1e6,
        "p99_ms": float(np.percentile(timings, 99)) / 1e6,
    }


def compare(original, folded, samples, repeats, thread_counts):
    """Measure both graphs with the same inputs, session options and timing boundaries."""
    results = []
    reference = None
    for optimize in (False, True):
        for threads in thread_counts:
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize
                else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            sessions = [
                ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
                for model in (str(original), folded)
            ]
            if reference is None:
                feeds = _benchmark_feeds(sessions[0])
                reference = [sessions[0].run(None, feed) for feed in feeds]
            durations = [[], []]
            for run in range(repeats):
                # Alternate which graph goes first so order does not favor one variant.
                for index in (run % 2, 1 - run % 2):
                    durations[index].append(_measure_session(sessions[index], feeds, reference, samples))
            for variant, runs in zip(("original", "folded"), durations):
                results.append({
                    "variant": variant, "ort_optimizations": optimize, "threads": threads,
                    "median_ms": median(run["median_ms"] for run in runs),
                    "p99_ms": median(run["p99_ms"] for run in runs),
                    "runs": runs,
                })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("folded", type=Path)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if min(args.samples, args.repeats, *args.threads) < 1:
        parser.error("Samples, repeats and thread counts must be positive")
    if args.folded.exists():
        parser.error("Choose a new output path")
    model = onnx.load(args.original)
    onnx.checker.check_model(model)
    count = fold_conv_batchnorm(model)
    if not count:
        parser.exit(1, "No eligible Conv–BatchNormalization pairs found\n")
    onnx.checker.check_model(model)
    folded = model.SerializeToString()
    results = compare(args.original, folded, args.samples, args.repeats, args.threads)
    with args.folded.open("xb") as output:
        output.write(folded)
    print(json.dumps({
        "pairs_folded": count,
        "onnxruntime_version": ort.__version__,
        "samples_per_repeat": args.samples,
        "repeats": args.repeats,
        "settings": results,
        "lowest_p99": min(results, key=lambda row: row["p99_ms"]),
    }, indent=2))


if __name__ == "__main__":
    main()
