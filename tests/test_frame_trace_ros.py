"""Opt-in ROS integration test. Run on an isolated ROS domain."""

import io
import json
import os
import time

import pytest

from ab.bot.frame_trace import (
    frame_key,
    output_frame_key,
    read_observations,
    record_receipt,
    summarize_frames,
    write_observation,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("NN_BOT_RUN_ROS_TESTS") != "1",
    reason="Set NN_BOT_RUN_ROS_TESTS=1 in a sourced ROS 2 environment",
)


def _write_test_model(path):
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["image"], ["result"])],
        "frame_trace_test",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.save(model, path)


def _spin_until(executor, condition, description):
    deadline = time.monotonic() + 5.0
    while not condition() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
    assert condition(), f"Timed out waiting for {description}"


def test_ros_frame_trace_reports_injected_drops_and_deadline_misses(tmp_path):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray

    from ab.bot.port import InferenceNode

    model = tmp_path / "frame_trace.onnx"
    inference_log = tmp_path / "inference.jsonl"
    _write_test_model(model)
    deadline_ns = 500_000_000
    delayed = {3, 5}
    namespace = f"/nn_bot_trace_test_{os.getpid()}"
    source_topic = f"{namespace}/camera"
    input_topic = f"{namespace}/inference_input"
    result_topic = f"{namespace}/result"

    rclpy.init(args=[
        "--ros-args",
        "-p", f"model_path:={model}",
        "-p", f"results_dir:={tmp_path}",
        "-p", f"input_topic:={input_topic}",
        "-p", f"output_topic:={result_topic}",
        "-p", f"ready_topic:={namespace}/ready",
        "-p", "onnx_execution_mode:=cpu",
        "-p", "onnx_num_threads:=1",
        "-p", "trace_output_frames:=true",
        "-p", f"inference_trace_path:={inference_log}",
        "-p", "startup_drop_frames:=0",
        "-p", "warmup_runs:=1",
        "-p", "iterations:=4",
    ])
    executor = SingleThreadedExecutor()
    inference = fixture = None
    try:
        inference = InferenceNode()
        fixture = Node("frame_trace_fault_fixture", use_global_arguments=False)
        executor.add_node(inference)
        executor.add_node(fixture)
        source_pub = fixture.create_publisher(Image, source_topic, 10)
        relay_pub = fixture.create_publisher(Image, input_topic, 10)
        source_log = io.StringIO()
        receipt_log = io.StringIO()
        frames_by_key = {}
        forwarded = []
        results = []
        arrivals = []
        deliveries = []

        def relay(image):
            index = frames_by_key[frame_key(image)]
            arrivals.append(index)
            if index != 1:
                relay_pub.publish(image)
                forwarded.append(index)

        def receive(output):
            received_ns = fixture.get_clock().now().nanoseconds
            key = output_frame_key(output)
            results.append(frames_by_key[key])
            record_receipt(receipt_log, output, received_ns)

        fixture.create_subscription(Image, source_topic, relay, 10)
        delivery_topic = f"{namespace}/consumer"
        delivery_pub = fixture.create_publisher(Float32MultiArray, delivery_topic, 10)
        fixture.create_subscription(Float32MultiArray, delivery_topic, receive, 10)

        def deliver(output):
            index = frames_by_key[output_frame_key(output)]
            deliveries.append(index)
            if index == 7:
                return
            if index in delayed:
                time.sleep(0.65)
            delivery_pub.publish(output)

        fixture.create_subscription(Float32MultiArray, result_topic, deliver, 10)
        _spin_until(
            executor,
            lambda: source_pub.get_subscription_count()
            and relay_pub.get_subscription_count()
            and fixture.count_publishers(result_topic)
            and delivery_pub.get_subscription_count(),
            "ROS topic discovery",
        )

        def publish(index):
            image = Image(
                height=8, width=8, encoding="invalid" if index == 4 else "rgb8",
                step=24, data=bytes([index]) * 192,
            )
            image.header.frame_id = "test_camera"
            image.header.stamp = fixture.get_clock().now().to_msg()
            key = frame_key(image)
            frames_by_key[key] = index
            source_pub.publish(image)
            write_observation(source_log, key)
            _spin_until(executor, lambda: index in arrivals, f"source frame {index}")
            if index == 4:
                _spin_until(executor, lambda: inference.failed_frames == 1, "inference failure")
            elif index == 7:
                _spin_until(executor, lambda: index in deliveries, "result before output drop")
            elif index != 1:
                _spin_until(executor, lambda: index in results, f"consumer result {index}")

        # Frame zero warms up inference. Only frames one through seven are scored.
        for index in range(8):
            publish(index)

        executor.remove_node(inference)
        inference.destroy_node()
        inference = None
        report = summarize_frames(
            map(json.loads, source_log.getvalue().splitlines()[1:]),
            map(json.loads, receipt_log.getvalue().splitlines()),
            deadline_ns,
            inference_rows=read_observations(inference_log),
        )
        assert forwarded == [0, 2, 3, 4, 5, 6, 7]
        assert deliveries == [0, 2, 3, 5, 6, 7]
        assert results == [0, 2, 3, 5, 6]
        assert report["source_frames"] == 7
        assert report["received_frames"] == 4
        assert report["missing_frames"] == 3
        assert report["late_received_frames"] == 2
        assert report["deadline_missed_frames"] == 5
        assert report["max_consecutive_missed_frames"] == 3
        assert report["camera_to_consumer_latency_ns"]["max"] >= 650_000_000
        assert report["inference_stages"] == {
            "arrived_frames": 6,
            "published_frames": 5,
            "missing_before_inference": 1,
            "missing_before_publish": 1,
            "missing_after_publish": 1,
        }
    finally:
        executor.shutdown()
        for node in (fixture, inference):
            if node is not None:
                node.destroy_node()
        rclpy.try_shutdown()
