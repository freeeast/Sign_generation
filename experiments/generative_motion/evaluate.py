"""Compare angular dynamics, diversity and coarse training-set neighbors."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from experiments.generative_motion.motion import GROUPS, fit_rotations, hand_mean, split_rows
from src.shared_motion import geodesic, increments, load_predictor, six


def features(matrix):
    # Compare length-normalized local rotation tracks at eight evenly spaced times.
    indices = np.linspace(0, len(matrix) - 1, 8, dtype=int)
    return six(matrix[indices]).flatten().numpy()


def report(sequences, fps):
    rows = {}
    for name, joints in GROUPS.items():
        group = [matrix[:, joints] for matrix in sequences]
        velocity = [np.rad2deg(increments_for_group(matrix)) * fps for matrix in group]
        rows[name] = {
            "angular_speed_median_deg_s": float(np.median(np.concatenate([np.linalg.norm(v, axis=-1).ravel() for v in velocity]))),
            "angular_speed_p95_deg_s": float(np.percentile(np.concatenate([np.linalg.norm(v, axis=-1).ravel() for v in velocity]), 95)),
            "angular_acceleration_p95_deg_s2": float(np.percentile(np.concatenate([np.linalg.norm(np.diff(v, axis=0) * fps, axis=-1).ravel() for v in velocity]), 95)),
            "angular_jerk_p95_deg_s3": float(np.percentile(np.concatenate([np.linalg.norm(np.diff(v, n=2, axis=0) * fps**2, axis=-1).ravel() for v in velocity]), 95)),
            "steps_over_90_deg_fraction": float(np.mean(np.concatenate([np.linalg.norm(v, axis=-1).ravel() > 90 * fps for v in velocity]))),
            "joints_below_1deg_mean_speed_fraction": float(np.mean(np.stack([np.linalg.norm(v, axis=-1).mean(axis=0) for v in velocity]) < 1)),
            "sequence_pairwise_mean_deg": float(np.mean([geodesic(a, b).mean().item() for a, b in combinations(group, 2)])) if len(group) > 1 else None,
        }
        rows[name]["rotation_det_error_max"] = max(float((torch.linalg.det(x) - 1).abs().max()) for x in group)
    return rows


def increments_for_group(matrix):
    from scipy.spatial.transform import Rotation
    relative = matrix[:-1].transpose(-1, -2) @ matrix[1:]
    return Rotation.from_matrix(relative.numpy().reshape(-1, 3, 3)).as_rotvec().reshape(len(relative), -1, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "asset", "samples", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--gru-checkpoint", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--max-reference", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    means = hand_mean(args.asset)
    rows = split_rows(args.dataset)
    metadata = json.loads((args.samples / "sampling.json").read_text(encoding="utf-8"))
    frames, fps = metadata["exported_frames"], metadata["fps"]
    if frames < 4 or args.max_reference < 1:
        parser.error("At least four frames and a positive reference count are required")
    generated = [
        torch.from_numpy(fit_rotations(args.samples, {"path": f"seed_{seed}.npz", "frames": frames}, means, fps))
        for seed in metadata["seeds"]
    ]
    rng = np.random.default_rng(args.seed)
    candidates = [row for row in rows[args.split] if int(row["frames"]) >= frames]
    if not candidates:
        raise ValueError("No reference clips long enough for this duration")
    reference, reference_ids = [], []
    for index in rng.permutation(len(candidates))[:args.max_reference]:
        row = candidates[index]
        full = fit_rotations(args.dataset, row, means, fps)
        start = int(rng.integers(0, len(full) - frames + 1))
        reference.append(torch.from_numpy(full[start:start + frames]))
        reference_ids.append({"clip": row["clip"], "start": start})
    train_descriptors = []
    for row in rows["train"]:
        if int(row["frames"]) < frames:
            continue
        full = torch.from_numpy(fit_rotations(args.dataset, row, means, fps))
        for start in range(0, len(full) - frames + 1, frames // 2):
            train_descriptors.append(features(full[start:start + frames]))
    if not train_descriptors:
        raise ValueError("No training windows of the requested duration for nearest neighbors")

    def neighbors(sequences):
        bank = np.stack(train_descriptors)
        values = [float(np.mean((bank - features(seq))**2, axis=1).min()**.5) for seq in sequences]
        return {"median_rot6d_rmse": float(np.median(values)), "individual_rot6d_rmse": values}

    result = {
        "split": args.split, "frames": frames, "fps": fps, "sampling": metadata,
        "reference_windows": reference_ids, "train_neighbor_windows": len(train_descriptors),
        "metrics": {"generated": report(generated, fps), "nsa": report(reference, fps)},
        "nearest_train": {"generated": neighbors(generated), "nsa": neighbors(reference)},
    }
    if args.gru_checkpoint:
        if fps != 24 or frames < 8:
            raise ValueError("Legacy GRU comparison expects 24 fps and at least eight frames")
        model, checkpoint = load_predictor(args.gru_checkpoint)
        context, horizon = checkpoint["context"], frames // 2
        prior = []
        for row in candidates:
            if len(prior) == args.max_reference:
                break
            if int(row["frames"]) < 2 * (context + horizon):
                continue
            rotations = fit_rotations(args.dataset, row, means, fps)[:2 * (context + horizon):2]
            prior.append((torch.from_numpy(rotations), torch.from_numpy(increments(rotations))))
        if not prior:
            raise ValueError("No eligible clips with legacy GRU history")
        hold, forecast, reference_12 = [], [], []
        with torch.inference_mode():
            for matrix, velocity in prior:
                output, _ = model.rollout(matrix[:context][None], velocity[:context][None], horizon)
                forecast.append(output[0])
                hold.append(matrix[context - 1:context].expand(horizon, -1, -1, -1))
                reference_12.append(matrix[context:])
        result["legacy_forecast_comparison_12hz"] = {
            "windows": len(prior), "gru": report(forecast, 12), "hold": report(hold, 12),
            "nsa": report(reference_12, 12), "generated": report([seq[:2 * horizon:2] for seq in generated], 12),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluated {len(generated)} samples and {len(reference)} {args.split} windows")


if __name__ == "__main__":
    main()
