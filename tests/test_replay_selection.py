import pytest

from ab.bot.replay_selection import compare_selectors


def test_selection_uses_only_calibration_and_counts_missing_and_late_on_holdout():
    def run(variant, round_number, host_p99, missing, late):
        return {
            "variant": variant, "threads": 1, "round": round_number, "condition": "baseline",
            "frames": 100, "deadline_misses": missing + late, "missing": missing, "late": late,
            "host_p99_ms": host_p99, "consumer_p99_ms": host_p99 + 2,
        }

    runs = [
        run("fast", 1, 1, 3, 0), run("steady", 1, 5, 0, 0),
        run("fast", 2, 6, 0, 0), run("steady", 2, 2, 5, 1),
    ]
    report = compare_selectors(runs, [1], [2])
    assert report["lowest_host_p99"]["variant"] == "fast"
    assert report["fewest_deadline_misses"]["variant"] == "steady"
    assert report["fewest_deadline_misses"]["holdout"]["misses"] == 6
    with pytest.raises(ValueError, match="disjoint"):
        compare_selectors(runs, [1], [1, 2])
    with pytest.raises(ValueError, match="one replay"):
        compare_selectors(runs[:-1], [1], [2])
