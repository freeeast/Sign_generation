"""Freeze source selection, pair exact timestamps and fit the existing proxy camera."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.stage4_geometry import evaluate, reference_hands
from src.run_real_validation import observations
from src.virtual_pose import model_sha, virtual_inputs


def select(inventory, annotations, cfg, old):
    seen = {row["video_id"] for row in old}
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        lookup = {r["SENTENCE_NAME"]: r for r in csv.DictReader(stream, delimiter="\t")}
    inventory = [line.split() for line in inventory.read_text().splitlines() if len(line.split()) == 2]
    selected = []
    for clip, n in sorted(inventory, key=lambda row: (int(row[1]), row[0])):
        n = int(n)
        if not cfg["minimum_frames"] <= n <= cfg["maximum_frames"] or clip not in lookup:
            continue
        row = lookup[clip]
        if row["VIDEO_ID"] in seen:
            continue
        selected.append(dict(clip_id=clip, video_id=row["VIDEO_ID"], video_name=row["VIDEO_NAME"],
                             split="val", evaluation_partition="new_source", reference=row["SENTENCE"],
                             original_start_s=float(row["START"]), original_end_s=float(row["END"]),
                             fit_frames=n, rgb_frames=n, rgb_fps=24))
        seen.add(row["VIDEO_ID"])
    return selected[:cfg["source_count"]]


def prepare(row, args):
    clip = row["clip_id"]
    data_dir = args.output / "data"
    target = data_dir / "paired" / f"{clip}.npz"
    real, scores, wh = observations(data_dir / "real_pose", row)
    with np.load(data_dir / "real_pose" / f"{clip}.npz", allow_pickle=False) as pose:
        times = pose["timestamps_s"]
        frame_time_max_error_ms = float(np.max(np.abs(times - np.arange(len(real)) / 24)) * 1000)
        if abs(float(pose["fps"])-24) > .01 or frame_time_max_error_ms > 3:
            raise ValueError(f"{clip}: timestamps disagree with the validated 24 fps frame convention")
    motion = args.motions / f"{clip}.npz"
    rgb = args.videos / f"{clip}.mp4"
    points, variants, camera, quality = virtual_inputs(motion, args.asset, real, scores, wh, "model", "cpu")
    xy = variants["proxy_matched_constant1"][0]
    detector = json.loads((data_dir / "real_pose" / f"{clip}.json").read_text())
    rgb_sha256 = hashlib.sha256(rgb.read_bytes()).hexdigest()
    if rgb_sha256 != detector["sha256"]:
        raise ValueError(f"{clip}: detector cache belongs to another RGB file")
    mp, count = reference_hands(rgb, args.mp_model, real, scores, wh)
    np.savez_compressed(target, original3d=points, original2d=xy, camera=camera, mp=mp)
    base = evaluate(xy, mp, wh, slice(1,None,2))
    source = json.loads(motion.with_suffix(".json").read_text())
    result = dict(**row, **quality, dimensions_px=list(wh), rgb_sha256=rgb_sha256,
                  nsa_members_sha256=source["archive_sha256"], frame_time_max_error_ms=frame_time_max_error_ms,
                  mp_total_detected=count, mp_odd_associated=int(np.isfinite(mp[1::2, [91,112], 0]).sum()),
                  mp_odd_error_px=base, rtmw_pose_coverage=detector["visible_fractions"])
    target.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    print(clip, len(points), wh, result["mp_odd_associated"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "annotations", "asset", "motions", "videos", "mp-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = json.loads(Path("configs/stage6.json").read_text())["selection"]
    old = json.loads(Path("configs/validation_pairs.json").read_text())
    rows = select(args.inventory, args.annotations, cfg, old)
    if len(rows) != cfg["source_count"]:
        raise ValueError("Selected inventory does not contain the configured number of new sources")
    expected_sha = json.loads(Path("experiments/E1_real/geometry_qc.json").read_text())["asset_sha256"]
    if model_sha(args.asset) != expected_sha:
        raise ValueError("The prior-stage SMPL-X asset is required")
    data_dir = args.output / "data"
    (data_dir / "paired").mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(json.dumps(rows, indent=2)+"\n")
    for row in rows:
        if not args.clips or row["clip_id"] in args.clips:
            prepare(row, args)


if __name__ == "__main__":
    main()
