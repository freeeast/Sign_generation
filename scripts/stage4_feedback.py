"""Measure frozen feedback on cached and new local 3D correction candidates."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np

from src.evaluate_feedback import write_csv
from src.probe_3d import ROTATION_GROUPS, candidates, regular_axes, rotate, finger_observations
from src.run_real_validation import load_manifest, observations
from src.smplx_to_pose import forward
from src.stage4_features import FrozenFeedback, distances
from src.unisign import UniSign
from src.virtual_pose import apply_affine, load_fit, pinhole, model_sha
from scripts.stage4_geometry import evaluate, WRISTS


def make_candidates(fit, settings):
    families = settings["families"]
    for family, name, dose, _, params in candidates(fit, seed=settings["fit_seed"]):
        if family in families and not name.startswith("restored"):
            yield family, name, dose, params
    window = np.sin(np.linspace(0, np.pi, len(fit["global_orient"]))) ** 2
    for family in families:
        group = ROTATION_GROUPS[family]
        corrupt = regular_axes(group, settings["fit_seed"])
        for seed in settings["tangent_seeds"][1:]:
            other = regular_axes(group, seed)
            changed = dict(fit)
            for field, ids in group.items():
                track = rotate(fit[field], ids, corrupt[field], 16, window)
                changed[field] = rotate(track, ids, other[field], 8, window)
            yield family, f"tangent_{seed}", 8, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "checkpoint", "mt5", "asset", "motions", "stage4"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    alt = args.stage4 / "geometry"
    projected = args.stage4 / "projection_cache"
    projected.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path("configs/stage4.json").read_text())
    audit = json.loads(Path("experiments/E1_real/geometry_qc.json").read_text())
    if audit["asset_sha256"] != model_sha(args.asset):
        raise ValueError("Asset differs from cached G1")
    feedback = FrozenFeedback(UniSign(args.repo, args.checkpoint, args.mt5, args.device, args.threads))
    combined = []
    for row in load_manifest(Path("configs/validation_pairs.json")):
        if not row["fit_frames"]:
            continue
        clip = row["clip_id"]
        fit = load_fit(args.motions / f"{clip}.npz")
        xy, score, wh = observations(Path("experiments/E3/real_pose"), row)
        with np.load(alt / f"{clip}.npz") as pose:
            reference, shift = pose["mp"], pose["shift"]
        with np.load(Path("experiments/E1_real") / clip / "fk_and_projection.npz") as pose:
            baseline, camera, p3 = pose["keypoints"], pose["camera"], pose["landmarks3d"]
        ref = feedback.extract(xy, score)
        coverage = int(np.isfinite(reference[3::4, WRISTS, 0]).sum())
        coverage_frames = int(np.isfinite(reference[3::4, WRISTS, 0]).any(axis=1).sum())
        staged = []
        projections = {}
        with (Path("experiments/E3_5") / f"{clip}_local_probes.csv").open(newline="") as handle:
            previous = {(r["family"], r["case"]): r for r in csv.DictReader(handle)}

        def score_pose(family, name, dose, pose, points):
            gid = "handshape" if family != "arm_trajectory" else "tips"
            for condition in ("baseline", "anchor_shift", "confidence07"):
                if condition != "baseline" and (name not in {"dose8", "original", "corrupted16", "alternative_from16"}):
                    continue
                test_pose = pose + shift[:, None] if condition == "anchor_shift" else pose
                conf = np.full(score.shape, .7 if condition == "confidence07" else 1., dtype=np.float32)
                cached = previous.get((family, name)) if condition == "baseline" else None
                measured = feedback.extract(test_pose, conf, None if cached else row["reference"])
                m = finger_observations(test_pose, xy, score, wh)
                mp_metrics = evaluate(test_pose, reference, wh, slice(3, None, 4))
                geom = "tip" if gid == "tips" else gid
                staged.append(dict(clip=clip, split="cal" if clip in settings["calibration"] else "eval",
                                   family=family, case=name, dose_deg=dose, condition=condition,
                                   nll=float(cached["nll"]) if cached else measured["nll"],
                                   nll_source="E3.5 cache" if cached else "new",
                                   geometry_A=m[f"{geom}_geometry_A_px"], geometry_B=m[f"{geom}_geometry_B_px"],
                                   mp_B=mp_metrics[gid], mp_B_hands=coverage, mp_B_frames=coverage_frames,
                                   **distances(measured, ref)))
            projections[family + "__" + name] = np.asarray(pose, dtype=np.float32)
            projections[family + "__" + name + "__3d"] = np.asarray(points, dtype=np.float32)

        score_pose("baseline", "original", 0, baseline, p3)
        with tempfile.TemporaryDirectory() as scratch:
            temp = Path(scratch) / "input.npz"
            for family, name, dose, parameters in make_candidates(fit, settings):
                np.savez(temp, **parameters)
                points, _ = forward(temp, args.asset, "neutral", "model", 64, args.device, False)
                pose = apply_affine(pinhole(points, fit["cam_trans"]), camera)
                score_pose(family, name, dose, pose, points)
        np.savez_compressed(projected / f"{clip}.npz", **projections)
        write_csv(args.stage4 / f"{clip}_feedback.csv", staged)
        combined.extend(staged)
        print(clip, len(staged), "MediaPipe hand instances on B", coverage, flush=True)
    write_csv(args.stage4 / "feedback.csv", combined)


if __name__ == "__main__":
    main()
