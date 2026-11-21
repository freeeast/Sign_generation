"""Export matching-duration held-out NSA, Hold and GRU comparison clips."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.generative_motion.motion import export_smplx, fit_rotations, hand_mean, split_rows
from src.shared_motion import increments, load_predictor, six


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "asset", "gru-checkpoint", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()
    if args.frames < 4 or args.frames % 2:
        parser.error("--frames must be even and at least four")
    torch.set_num_threads(2)
    means = hand_mean(args.asset)
    model, checkpoint = load_predictor(args.gru_checkpoint)
    context, horizon = checkpoint["context"], args.frames // 2
    rows = split_rows(args.dataset)["val"]
    row = next((row for row in rows if int(row["frames"]) >= 2 * (context + horizon)), None)
    if row is None:
        raise ValueError("No validation clip long enough for the requested horizon")
    rot = fit_rotations(args.dataset, row, means, 24)
    down = torch.from_numpy(rot[:2 * (context + horizon):2])
    with torch.inference_mode():
        vel = torch.from_numpy(increments(down))
        forecast, _ = model.rollout(down[:context][None], vel[:context][None], horizon)
    hold = down[context - 1:context].expand(horizon, -1, -1, -1)
    args.output.mkdir(parents=True, exist_ok=True)
    future = torch.from_numpy(rot[2 * context:2 * context + args.frames])
    for name, matrix in (("NSA_validation", future), ("GRU_forecast", forecast[0]), ("Hold", hold)):
        if name != "NSA_validation":
            matrix = matrix.repeat_interleave(2, dim=0)
        export_smplx(six(matrix).flatten(start_dim=1), means, 24, args.output / f"{name}.npz")
    (args.output / "comparison.json").write_text(json.dumps({
        "clip": row["clip"], "seed_frames_12hz": context, "compared_frames_24hz": args.frames,
        "note": "GRU/Hold use true history; 12 Hz frames repeated only for playback.",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
