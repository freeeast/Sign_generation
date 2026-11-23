"""Launch the pinned SignSparK modes with explicit text and keyframe controls."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys

PIN = "916402d0ecd6322fb4a9e5f513229a94a57bf4b6"
STREAMS = {"hand": 90, "body": 60, "face": 56}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mode", choices=("A", "B", "C", "D"), required=True)
    parser.add_argument("--stream", choices=STREAMS, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    repo, checkpoint, output = (p.resolve() for p in (args.repo, args.checkpoint, args.output))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != PIN:
        parser.error(f"Use audited upstream commit {PIN}")
    imports = ("torch", "hydra", "torchdiffeq", "wandb", "transformers", "lmdb", "blobfile")
    missing = [name for name in imports if importlib.util.find_spec(name) is None]
    if missing:
        parser.error("Install the pinned upstream environment. Missing: " + ", ".join(missing))
    if not checkpoint.is_file():
        parser.error("Provide the release checkpoint for the selected stream")
    if args.frames < 16 or args.frames % 16 or args.steps < 1:
        parser.error("Use a positive step count and a frame count divisible by 16")
    # Upstream A–C require a reference LMDB. D uses a constant requested length.
    if args.mode != "D" and (args.data_root is None or not args.data_root.is_dir()):
        parser.error("A–C need the released dataset directory")
    data_root = None if args.data_root is None else args.data_root.resolve()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from basic_utils import args_to_dict, create_model_and_flow
    from signspark.unet_model import Wilor_UNET
    from signspark.utils import dist_util

    flags = [f"data_v2={args.stream}", "data_v2/common@common=inference",
             f"eval.model_path={checkpoint}", f"eval.ode_stepnum={args.steps}",
             f"seed={args.seed}", "eval.batch_size=1", "eval.max_samples=1",
             "eval.text_guidance_scale=-1"]
    if args.mode != "D":
        os.environ["DATA_ROOT"] = str(data_root)
        os.environ["OUTPUT_DIR"] = str(output)
        if args.mode == "B":
            # Preserve the model and observed keyframes, suppress text before inference.
            forward = Wilor_UNET.forward

            def without_text(self, x, timesteps, y=None, **kwargs):
                return forward(self, x, timesteps, dict(y, uncond=True), **kwargs)

            Wilor_UNET.forward = without_text
        if args.mode == "C":
            flags.append("eval.keyframes_mask_all=True")
        sys.argv = ["sample.py", "test_data=[How2Sign]", f"eval.note=mode_{args.mode}", *flags]
        runpy.run_path("sample.py", run_name="__main__")
        return

    import numpy as np
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
        cfg = compose(config_name="default", overrides=flags)
    model, flow = create_model_and_flow(**args_to_dict(cfg, cfg.keys()))
    model.load_state_dict(dist_util.load_state_dict(checkpoint, map_location="cpu"), strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    batch = 2 if args.stream == "hand" else 1
    shape = (batch, STREAMS[args.stream], args.frames)
    random = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.randn(shape, device=args.device, generator=random)
    empty = torch.zeros_like(noise)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    labels = {"mask": torch.ones(batch, 1, args.frames, dtype=torch.bool, device=args.device),
              "text": [""] * batch, "uncond": True}
    generated, _ = flow.decode(model, noise=noise, keyframe_mask=mask, x_embed=empty,
                               model_kwargs={"y": labels, "obs_x0": empty, "obs_mask": mask},
                               ode_package=cfg.eval.sampler, ode_stepnum=args.steps)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{args.stream}_seed_{args.seed}.npz",
                        upstream_features=generated.cpu().numpy(), length=args.frames, seed=args.seed)


if __name__ == "__main__":
    main()
