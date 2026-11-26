"""Correlate publisher and consumer observations for a single camera stream."""

import argparse
import json
import math
from pathlib import Path

from ab.bot.runtime import message_stamp_ns, timing_stats

TRACE_MARKER = "|nn_bot_frame="


def frame_key(image):
    return (image.header.frame_id, message_stamp_ns(image))


def tag_output(output, key):
    if not output.layout.dim:
        raise ValueError("Frame tracing requires a tensor output with at least one dimension")
    output.layout.dim[0].label += TRACE_MARKER + json.dumps(key, separators=(",", ":"))


def output_frame_key(output):
    if not output.layout.dim:
        return None
    _, marker, value = output.layout.dim[0].label.partition(TRACE_MARKER)
    return tuple(json.loads(value)) if marker else None


def write_observation(stream, key, **fields):
    stream.write(json.dumps({"frame_key": key, **fields}, separators=(",", ":")) + "\n")


def record_receipt(stream, message, received_ns):
    """Call at consumer callback entry, using the camera's clock domain."""
    key = output_frame_key(message)
    if key is not None:
        write_observation(stream, key, received_ns=received_ns)


def summarize_frames(source_rows, consumer_rows, deadline_ns, *, inference_rows=None):
    """Use source log order as the denominator and the earliest receipt per frame."""
    source_keys = [tuple(row["frame_key"]) for row in source_rows]
    if not source_keys or len(set(source_keys)) != len(source_keys):
        raise ValueError("Source trace must contain at least one frame and unique frame keys")
    if any(key[1] <= 0 for key in source_keys):
        raise ValueError("Source trace requires positive capture timestamps")
    receipts = {}
    for row in consumer_rows:
        key = tuple(row["frame_key"])
        timestamp = row["received_ns"]
        receipts[key] = min(timestamp, receipts.get(key, timestamp))

    samples = []
    missing = late = streak = longest_streak = 0
    for key in source_keys:
        receipt = receipts.get(key)
        if receipt is None:
            missing += 1
            missed = True
        else:
            latency = receipt - key[1]
            if latency < 0:
                raise ValueError("Receipt precedes capture. Verify camera and consumer clocks.")
            samples.append(latency)
            missed = latency > deadline_ns
            late += int(missed)
        streak = streak + 1 if missed else 0
        longest_streak = max(streak, longest_streak)

    source_span_ns = source_keys[-1][1] - source_keys[0][1]
    source_rate_hz = (
        (len(source_keys) - 1) * 1_000_000_000 / source_span_ns
        if source_span_ns > 0 else None
    )
    latency = timing_stats(samples) if samples else None
    if latency is not None:
        latency.pop("samples")
    report = {
        "source_frames": len(source_keys),
        "observed_source_rate_hz": source_rate_hz,
        "received_frames": len(samples),
        "missing_frames": missing,
        "late_received_frames": late,
        "deadline_missed_frames": missing + late,
        "max_consecutive_missed_frames": longest_streak,
        "deadline_ns": deadline_ns,
        "camera_to_consumer_latency_ns": latency,
    }
    if inference_rows is not None:
        source_window = set(source_keys)
        stages = {"arrival": set(), "publish": set()}
        for row in inference_rows:
            key = tuple(row["frame_key"])
            if key in source_window:
                stages[row["stage"]].add(key)
        arrived = stages["arrival"]
        published = stages["publish"]
        received = receipts.keys() & source_window
        if not (received <= published <= arrived):
            raise ValueError(
                "Frame observations are missing upstream events. Verify that the "
                "inference and consumer logs cover the same source window."
            )
        report["inference_stages"] = {
            "arrived_frames": len(arrived),
            "published_frames": len(published),
            "missing_before_inference": len(source_window - arrived),
            "missing_before_publish": len(arrived - published),
            "missing_after_publish": len(published - received),
        }
    return report


def read_observations(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def listen(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Float32MultiArray

    rclpy.init(args=[])
    node = Node("nn_bot_frame_trace", parameter_overrides=[
        Parameter("use_sim_time", value=args.use_sim_time)
    ])
    try:
        with args.consumer_log.open("x", encoding="utf-8") as stream:
            def on_result(message):
                received_ns = node.get_clock().now().nanoseconds
                record_receipt(stream, message, received_ns)

            node.create_subscription(Float32MultiArray, args.topic, on_result, 10)
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("listen", help="Record receipts at a diagnostic subscriber")
    capture.add_argument("topic")
    capture.add_argument("consumer_log", type=Path)
    capture.add_argument("--use-sim-time", action="store_true")
    summary = commands.add_parser("summarize", help="Compare a source frame window to receipt logs")
    summary.add_argument("source_log", type=Path)
    summary.add_argument("consumer_log", type=Path)
    summary.add_argument("--inference-log", type=Path, help="Inference arrival and publish trace")
    summary.add_argument("--deadline-ms", type=float, required=True)
    summary.add_argument("--first-frame", type=int, required=True, help="First source log row, one-based")
    summary.add_argument("--last-frame", type=int, required=True, help="Last source log row, inclusive")
    args = parser.parse_args()
    if args.command == "listen":
        try:
            listen(args)
        except OSError as error:
            parser.error(str(error))
        return

    if not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0:
        parser.error("--deadline-ms must be finite and positive")
    try:
        sources = list(read_observations(args.source_log))
        if not 1 <= args.first_frame <= args.last_frame <= len(sources):
            parser.error("Select an inclusive frame range within the source log")
        if len({tuple(row["frame_key"]) for row in sources}) != len(sources):
            parser.error("Source frame keys must be unique throughout the log")
        report = summarize_frames(
            sources[args.first_frame - 1:args.last_frame],
            read_observations(args.consumer_log),
            round(args.deadline_ms * 1_000_000),
            inference_rows=read_observations(args.inference_log) if args.inference_log else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
