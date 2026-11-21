"""Run Uni-Sign inference and paired, within-clip feedback diagnostics."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .perturb_motion import perturbations
from .pose import load_pose
from .unisign import UniSign, cosine


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("Empty manifest")
    model = UniSign(args.repo, args.checkpoint, args.mt5, args.device, args.threads)
    output = []
    clean_emb = {}
    clip_by_id = {row["clip_id"]: row for row in rows}
    summaries = defaultdict(list)
    for current, row in enumerate(rows):
        clip = row["clip_id"]
        path = Path(row.get("pose", clip + args.pose_suffix))
        if not path.is_absolute():
            path = args.pose_dir / path
        xy, score = load_pose(path, trusted_pickle=args.trusted_pickle)
        cases = perturbations(xy, score, np.random.default_rng(args.seed + current)) if args.perturb else [
            ("clean", xy, score)]
        for condition, positions, scores in cases:
            start = perf_counter()
            targets = [row["reference"]]
            paired = [i for i in range(len(rows)) if
                      args.perturb and rows[i]["reference"] != row["reference"] and
                      rows[i]["clip_id"][:11] != clip[:11]] if condition == "clean" else []
            targets.extend(rows[i]["reference"] for i in paired)
            res = model.infer(positions, scores, targets, max_frames=args.max_frames,
                              beams=args.beams if condition == "clean" else 0)
            nll = res["target_nll"][0]
            if condition == "clean":
                clean_nll = nll
                clean_emb[clip] = res["embedding"]
            entry = dict(clip_id=clip, role=row.get("role", "pilot"), variant=condition,
                         pose_clip_id=clip, frames_input=len(positions), frames_model=res["frames"],
                         reference=row["reference"], prediction=res["prediction"],
                         target_nll=nll, delta_nll=nll - clean_nll,
                         cosine_to_clean=cosine(res["embedding"], clean_emb[clip]),
                         wall_seconds=perf_counter() - start)
            output.append(entry)
            if entry["role"] != "known_output_positive_control":
                summaries[condition].append(entry)
            for idx, wrong_nll in zip(paired, res["target_nll"][1:]):
                # This motion is a negative for the other sentence. Delay the delta
                # until the correct motion for that sentence is also available.
                output.append(dict(entry, clip_id=rows[idx]["clip_id"], variant="mismatched_motion",
                                   role=rows[idx].get("role", "pilot"), reference=rows[idx]["reference"],
                                   prediction=None, wall_seconds=None,
                                   target_nll=wrong_nll, delta_nll=None, cosine_to_clean=None))
            print(clip, condition, round(nll, 4), res["prediction"] or "", flush=True)
    clean_by_id = {row["clip_id"]: row for row in output if row["variant"] == "clean"}
    # Use one mismatched motion per target, selecting by similar length without scores.
    negatives_by_id = defaultdict(list)
    for row in output:
        if row["variant"] == "mismatched_motion":
            negatives_by_id[row["clip_id"]].append(row)
    output = [row for row in output if row["variant"] != "mismatched_motion"]
    for clip, negatives in negatives_by_id.items():
        matching = clean_by_id[clip]
        pilot_negatives = [negative for negative in negatives
                           if clip_by_id[negative["pose_clip_id"]].get("role") != "known_output_positive_control"]
        if not pilot_negatives:
            continue
        row = min(pilot_negatives, key=lambda candidate: (
            abs(candidate["frames_input"] - matching["frames_input"]), candidate["pose_clip_id"]))
        row["delta_nll"] = row["target_nll"] - matching["target_nll"]
        row["cosine_to_clean"] = cosine(clean_emb[row["pose_clip_id"]], clean_emb[clip])
        output.append(row)
        if row["role"] != "known_output_positive_control":
            summaries[row["variant"]].append(row)
    summary = []
    for condition, group in summaries.items():
        deltas = np.array([x["delta_nll"] for x in group])
        similarities = [x["cosine_to_clean"] for x in group if x["cosine_to_clean"] is not None]
        summary.append(dict(variant=condition, n=len(group),
                            mean_delta_nll=float(deltas.mean()), median_delta_nll=float(np.median(deltas)),
                            n_delta_above_1e_4=int((deltas > 1e-4).sum()),
                            min_delta_nll=float(deltas.min()), max_delta_nll=float(deltas.max()),
                            median_cosine=float(np.median(similarities)) if similarities else None))
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_clip.csv", output)
    if summary:
        write_csv(args.output / "summary.csv", summary)
    np.savez_compressed(args.output / "clean_embeddings.npz", **clean_emb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("repo", "checkpoint", "mt5", "manifest", "pose-dir", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--pose-suffix", choices=(".npz", ".pkl"), default=".npz")
    parser.add_argument("--trusted-pickle", action="store_true")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--beams", type=int, default=4, help="0 disables text generation")
    args = parser.parse_args()
    if args.beams < 0 or args.max_frames < 1 or args.threads < 0:
        parser.error("Use nonnegative beams/threads and positive max-frames")
    run(args)


if __name__ == "__main__":
    main()
