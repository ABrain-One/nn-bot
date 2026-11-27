"""Compare inference-latency and deadline-based selection on separate replay rounds."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
from statistics import median


def compare_selectors(runs, calibration_rounds, holdout_rounds):
    calibration = set(calibration_rounds)
    holdout = set(holdout_rounds)
    if not calibration or not holdout or calibration & holdout:
        raise ValueError("Calibration and holdout rounds must be nonempty and disjoint")

    groups = defaultdict(list)
    coverage = set()
    conditions = set()
    for run in runs:
        round_number = run["round"]
        if round_number not in calibration | holdout:
            continue
        config = (run["variant"], run["threads"])
        condition = run["condition"]
        key = (config, round_number, condition)
        if key in coverage:
            raise ValueError(f"Duplicate replay: {key}")
        coverage.add(key)
        conditions.add(condition)
        groups[config].append(run)
    if len(groups) < 2 or coverage != set(product(groups, calibration | holdout, conditions)):
        raise ValueError("Each configuration needs one replay per round and condition")

    def metrics(rows):
        frames = sum(row["frames"] for row in rows)
        misses = sum(row["deadline_misses"] for row in rows)
        return {
            "runs": len(rows), "frames": frames, "misses": misses,
            "missing": sum(row["missing"] for row in rows),
            "late": sum(row["late"] for row in rows),
            "miss_rate": misses / frames,
            "median_host_p99_ms": median(row["host_p99_ms"] for row in rows),
            "median_consumer_p99_ms": median(row["consumer_p99_ms"] for row in rows),
        }

    scores = {
        config: metrics([row for row in rows if row["round"] in calibration])
        for config, rows in groups.items()
    }
    fastest = min(scores, key=lambda config: (scores[config]["median_host_p99_ms"], config))
    deadline = min(scores, key=lambda config: (
        scores[config]["miss_rate"], scores[config]["median_consumer_p99_ms"], config
    ))

    def evaluate(config):
        rows = [row for row in groups[config] if row["round"] in holdout]
        return {
            "variant": config[0], "threads": config[1], "calibration": scores[config],
            "holdout": metrics(rows),
            "holdout_by_condition": {
                condition: metrics([row for row in rows if row["condition"] == condition])
                for condition in sorted(conditions)
            },
        }

    return {
        "calibration_rounds": sorted(calibration), "holdout_rounds": sorted(holdout),
        "lowest_host_p99": evaluate(fastest), "fewest_deadline_misses": evaluate(deadline),
        "calibration_candidates": [
            {"variant": variant, "threads": threads, **score}
            for (variant, threads), score in sorted(scores.items())
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Per-run JSON reports")
    parser.add_argument("--calibration", type=int, nargs="+", required=True)
    parser.add_argument("--holdout", type=int, nargs="+", required=True)
    args = parser.parse_args()
    try:
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
        report = compare_selectors(runs, args.calibration, args.holdout)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
