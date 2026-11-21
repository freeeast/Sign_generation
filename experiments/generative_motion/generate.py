"""Sample motion directly from Gaussian noise and export canonical SMPL-X arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.generative_motion.diffusion import Denoiser, GaussianDDPM
from experiments.generative_motion.motion import REPRESENTATION, export_smplx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state["representation"] != REPRESENTATION:
        raise ValueError("Unknown motion representation")
    cfg = state["config"]
    frames = cfg["frames"] if args.num_frames is None else args.num_frames
    if not 1 <= frames <= cfg["frames"] or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use unique seeds and a duration up to the checkpoint's trained frame count")
    torch.set_num_threads(cfg["threads"])
    device = torch.device(args.device)
    model = Denoiser(cfg["width"], cfg["depth"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    diffusion = GaussianDDPM(cfg["diffusion_steps"], device, cfg.get("prediction", "noise"),
                             cfg.get("velocity_weight", 0.0))
    mean, scale = state["mean"].to(device), state["scale"].to(device)
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in args.seeds:
        sample = diffusion.sample(model, cfg["frames"], seed) * scale + mean
        export_smplx(sample[:frames], state["hand_means"].numpy(), cfg["fps"], args.output / f"seed_{seed}.npz")
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "sampling.json").write_text(json.dumps({
        "trained_frames": cfg["frames"], "exported_frames": frames, "fps": cfg["fps"],
        "seeds": args.seeds, "checkpoint_sha256": digest, "checkpoint_step": state["step"],
        "representation": REPRESENTATION, "conditioning": "none", "device": args.device,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
