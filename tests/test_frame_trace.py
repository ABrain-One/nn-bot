import io
import json
from types import SimpleNamespace

import pytest

from ab.bot.frame_trace import main, record_receipt, summarize_frames, tag_output, write_observation


def test_frame_trace_counts_edge_losses_and_uses_first_receipt():
    frame_keys = [("camera", index * 100) for index in range(1, 6)]
    source_rows = [{"frame_key": key} for key in frame_keys]
    consumer_log = io.StringIO()
    for index, receipt in [(1, 260), (2, 320), (2, 360)]:
        message = SimpleNamespace(layout=SimpleNamespace(dim=[SimpleNamespace(label="dim_0")]))
        tag_output(message, frame_keys[index])
        record_receipt(consumer_log, message, receipt)
    report = summarize_frames(source_rows, map(json.loads, consumer_log.getvalue().splitlines()), 50)
    assert report["source_frames"] == 5
    assert report["observed_source_rate_hz"] == 10_000_000
    assert report["received_frames"] == 2
    assert report["missing_frames"] == 3
    assert report["late_received_frames"] == 1
    assert report["deadline_missed_frames"] == 4
    assert report["max_consecutive_missed_frames"] == 2
    assert report["camera_to_consumer_latency_ns"]["max"] == 60


def test_frame_trace_rejects_ambiguous_source_keys_and_negative_latency():
    source = {"frame_key": ("camera", 100)}
    with pytest.raises(ValueError, match="unique frame keys"):
        summarize_frames([source, source], [], 50)
    with pytest.raises(ValueError, match="precedes capture"):
        summarize_frames([source], [{**source, "received_ns": 99}], 50)


def test_frame_trace_cli_uses_the_same_source_window_at_every_stage(tmp_path, monkeypatch, capsys):
    source_path, inference_path, consumer_path = (
        tmp_path / name for name in ("source.jsonl", "inference.jsonl", "consumer.jsonl")
    )
    first, selected = ("camera", 100), ("camera", 200)
    with source_path.open("w") as stream:
        write_observation(stream, first)
        write_observation(stream, selected)
    with inference_path.open("w") as stream:
        write_observation(stream, selected, stage="arrival")
        write_observation(stream, selected, stage="arrival")
        write_observation(stream, selected, stage="publish")
    with consumer_path.open("w") as stream:
        write_observation(stream, first, received_ns=150)
        write_observation(stream, selected, received_ns=250)
    monkeypatch.setattr("sys.argv", [
        "frame_trace", "summarize", str(source_path), str(consumer_path),
        "--inference-log", str(inference_path), "--first-frame", "2",
        "--last-frame", "2", "--deadline-ms", "33",
    ])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["received_frames"] == 1
    assert report["inference_stages"] == {
        "arrived_frames": 1,
        "published_frames": 1,
        "missing_before_inference": 0,
        "missing_before_publish": 0,
        "missing_after_publish": 0,
    }


def test_frame_trace_rejects_missing_upstream_observations():
    source = {"frame_key": ("camera", 100)}
    with pytest.raises(ValueError, match="missing upstream events"):
        summarize_frames(
            [source], [{**source, "received_ns": 150}], 50,
            inference_rows=[{**source, "stage": "arrival"}],
        )
