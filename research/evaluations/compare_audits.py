"""Estimate paired changes over fixed inference seeds using audit CSV files."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_samples(path):
    with path.open(newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["domain"] == "generated"]
    return {int(row["sample"]): row for row in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline, candidate = load_samples(args.baseline), load_samples(args.candidate)
    if baseline.keys() != candidate.keys() or len(baseline) < 2:
        raise ValueError("Compare the same inference seeds")
    seeds = sorted(baseline)
    rng = np.random.default_rng(771)
    indices = rng.integers(0, len(seeds), (4000, len(seeds)))
    result = {}
    for field in ("left_step_p95_deg", "right_step_p95_deg", "left_jerk_p95_deg_s3",
                  "right_jerk_p95_deg_s3", "train_descriptor_nearest_rmse"):
        difference = np.array([float(candidate[i][field]) - float(baseline[i][field]) for i in seeds])
        result[field] = {
            "mean_delta": float(difference.mean()),
            "conditional_bootstrap_percentile_95": np.percentile(
                difference[indices].mean(axis=1), [2.5, 97.5]).tolist(),
        }
    args.output.write_text(json.dumps({"inference_seeds": len(seeds), "seed": 771,
                                       "differences": result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
