"""Source-disjoint failure audit, nearest training windows and optional SMPL-X FK checks."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
import torch

from experiments.generative_motion.evaluate import features
from experiments.generative_motion.motion import GROUPS, export_smplx, fit_rotations, hand_mean, split_rows
from src.shared_motion import six
from src.smplx_to_pose import forward


def stats(matrix, fps):
    result = {}
    for group, joints in GROUPS.items():
        rotations = matrix[:, joints]
        relative = rotations[:-1].swapaxes(-1, -2) @ rotations[1:]
        angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec()
        turns = np.rad2deg(angles).reshape(len(relative), -1, 3)
        speed = np.linalg.norm(turns, axis=-1)
        result[f"{group}_step_p95_deg"] = float(np.percentile(speed, 95))
        result[f"{group}_jump_fraction"] = float(np.mean(speed > 90))
        result[f"{group}_jerk_p95_deg_s3"] = float(np.percentile(
            np.linalg.norm(np.diff(turns, n=2, axis=0), axis=-1), 95) * fps**3)
    deviation = matrix.swapaxes(-1, -2) @ matrix - np.eye(3)
    result["rotation_max_orthogonality_error"] = float(np.abs(deviation).max())
    result["rotation_max_determinant_error"] = float(np.abs(np.linalg.det(matrix) - 1).max())
    return result


def fk_stats(matrix, means, asset, temp, fps):
    export_smplx(six(torch.from_numpy(matrix)).flatten(1), means, fps, temp)
    joints, mapped = forward(temp, asset.parent.parent, "neutral", "model", 32, "cpu", True)
    if not np.isfinite(joints).all() or not mapped[91:].all():
        raise ValueError("Invalid bilateral SMPL-X landmarks")
    measures = {}
    wrist_velocities = []
    for side, wrist in (("left", 91), ("right", 112)):
        tip = joints[:, [wrist + 4 * digit for digit in range(1, 6)]]
        wrist_xyz = joints[:, wrist]
        relative = tip - wrist_xyz[:, None]
        speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1) * fps * 1000
        wrist_velocities.append(np.linalg.norm(np.diff(wrist_xyz, axis=0), axis=-1))
        aperture = np.linalg.norm(tip[:, 0] - tip[:, -1], axis=-1)
        measures[f"{side}_tip_speed_median_mm_s"] = float(np.median(speed))
        measures[f"{side}_aperture_span_mm"] = float(np.ptp(aperture) * 1000)
        measures[f"{side}_wrist_torso_distance_p05_mm"] = float(np.percentile(
            np.linalg.norm(wrist_xyz - joints[:, [5, 6, 11, 12]].mean(axis=1), axis=-1), 5) * 1000)
    left, right = wrist_velocities
    if np.std(left) > 1e-8 and np.std(right) > 1e-8:
        measures["wrist_speed_synchrony"] = float(np.corrcoef(left, right)[0, 1])
    else:
        measures["wrist_speed_synchrony"] = 0.0
    return measures


def window_bank(dataset, rows, means, fps, frames):
    descriptors = []
    for row in rows:
        if int(row["frames"]) < frames:
            continue
        full = fit_rotations(dataset, row, means, fps)
        starts = list(range(0, len(full) - frames + 1, frames // 2))
        if starts[-1] != len(full) - frames:
            starts.append(len(full) - frames)
        descriptors.extend(features(torch.from_numpy(full[start:start + frames])) for start in starts)
    return np.stack(descriptors)


def manifold_coverage(reference, generated):
    # k=3 neighborhoods in a coarse unit-column rotation descriptor.
    radius = lambda values: np.partition(distance.cdist(values, values), 3, axis=1)[:, 3]
    cross = distance.cdist(generated, reference)
    return {
        "precision_proxy": float(np.mean(np.any(cross <= radius(reference)[None], axis=1))),
        "recall_proxy": float(np.mean(np.any(cross <= radius(generated)[:, None], axis=0))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "asset", "samples", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--fk-count", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    means = hand_mean(args.asset)
    rows = split_rows(args.dataset)
    meta = json.loads((args.samples / "sampling.json").read_text())
    frames, fps = meta["exported_frames"], meta["fps"]
    if frames < 4 or len(meta["seeds"]) < 4 or args.fk_count < 0:
        parser.error("Use at least four frames and four samples, and a nonnegative FK count")
    eligible = [r for r in rows[args.split] if int(r["frames"]) >= frames]
    rng = np.random.default_rng(456)
    reference = []
    generated = []
    for i in rng.permutation(len(eligible))[:32]:
        row = eligible[i]
        full = fit_rotations(args.dataset, row, means, fps)
        start = int(rng.integers(0, len(full) - frames + 1))
        reference.append((len(reference), full[start:start + frames]))
    for seed in meta["seeds"]:
        row = {"path": f"seed_{seed}.npz", "frames": frames}
        generated.append((seed, fit_rotations(args.samples, row, means, fps)))
    if len(reference) < 4:
        raise ValueError("At least four held-out windows are required")
    bank = window_bank(args.dataset, rows["train"], means, fps, frames)
    all_rows = []
    descriptors = []
    with TemporaryDirectory() as scratch:
        for domain, windows in (("reference", reference), ("generated", generated)):
            vectors = np.stack([features(torch.from_numpy(matrix)) for _, matrix in windows])
            nearest = distance.cdist(vectors, bank).min(axis=1) / vectors.shape[1]**.5
            descriptors.append(vectors)
            for index, ((seed, matrix), neighbor) in enumerate(zip(windows, nearest)):
                record = {"domain": domain, "sample": seed, **stats(matrix, fps),
                          "train_descriptor_nearest_rmse": float(neighbor)}
                if index < args.fk_count:
                    record.update(fk_stats(matrix, means, args.asset, Path(scratch) / "fit.npz", fps))
                all_rows.append(record)
    thresholds = {
        key: float(np.percentile([r[key] for r in all_rows if r["domain"] == "reference"], 95))
        for key in ("left_step_p95_deg", "right_step_p95_deg",
                    "left_jerk_p95_deg_s3", "right_jerk_p95_deg_s3")
    }
    for record in all_rows:
        record["jitter_outlier"] = any(record[k] > v for k, v in thresholds.items())
    summary = {"split": args.split, "sampling": meta, "thresholds": thresholds,
               "train_neighbor_windows": len(bank), **manifold_coverage(*descriptors)}
    for domain in ("reference", "generated"):
        group = [r for r in all_rows if r["domain"] == domain]
        summary[domain] = {"clips": len(group), "jitter_outliers": sum(r["jitter_outlier"] for r in group)}
        for key in group[0].keys() - {"domain", "sample", "jitter_outlier"}:
            values = [r[key] for r in group if key in r]
            summary[domain][key + "_median"] = float(np.median(values))
        summary[domain]["train_descriptor_near_exact_count"] = sum(
            r["train_descriptor_nearest_rmse"] < 1e-6 for r in group)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_sample.csv").open("w", newline="") as stream:
        keys = sorted(set().union(*(r.keys() for r in all_rows)))
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("precision_proxy", "recall_proxy", "generated")}))


if __name__ == "__main__":
    main()
