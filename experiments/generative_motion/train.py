"""Train or resume a shared unconditional model on source-separated SMPL-X windows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from experiments.generative_motion.diffusion import Denoiser
from experiments.generative_motion.flow import build_process
from experiments.generative_motion.motion import REPRESENTATION, Windows, hand_mean, split_rows


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def draw(windows, batch, rng, device):
    indices = torch.randint(len(windows), (batch,), generator=rng)
    return torch.stack([windows[index] for index in indices.tolist()]).to(device)


def statistics(windows):
    total = 0
    sums = 0
    squares = 0
    for clip in windows.clips:
        values = clip.double()
        total += len(values)
        sums += values.sum(dim=0)
        squares += values.square().sum(dim=0)
    mean = sums / total
    scale = (squares / total - mean.square()).clamp_min(0).sqrt().clamp_min(0.025)
    return mean.float(), scale.float()


@torch.inference_mode()
def validation(model, diffusion, windows, config, mean, scale, device):
    model.eval()
    seed = config["seed"] + 2
    index_rng = torch.Generator().manual_seed(seed)
    noise_rng = torch.Generator(device=device).manual_seed(seed)
    losses = []
    for _ in range(config["validation_batches"]):
        motion = draw(windows, config["batch_size"], index_rng, device)
        losses.append(diffusion.loss(model, (motion - mean) / scale, noise_rng).item())
    return sum(losses) / len(losses)


def run(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if cfg["width"] < 8 or cfg["width"] % 8 or cfg["frames"] < 4 or any(
        cfg[name] < 1 for name in ("depth", "diffusion_steps", "stride", "steps",
                                  "batch_size", "validate_every", "validation_batches", "threads")
    ):
        raise ValueError("Invalid model or training configuration")
    torch.set_num_threads(cfg["threads"])
    torch.manual_seed(cfg["seed"])
    device = torch.device(args.device)
    means = hand_mean(args.asset)
    rows = split_rows(args.dataset)
    train, val = (
        Windows(args.dataset, rows[split], means, cfg["frames"], cfg["stride"], cfg["fps"])
        for split in ("train", "val")
    )
    if not len(train) or not len(val):
        raise ValueError("No eligible fixed-length windows in train or validation")
    provenance = {
        "manifest_sha256": hashlib.sha256((args.dataset / "sample.csv").read_bytes()).hexdigest(),
        "asset_sha256": hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "experiments/generative_motion/train.py",
                "experiments/generative_motion/diffusion.py",
                "experiments/generative_motion/flow.py",
                "experiments/generative_motion/motion.py",
                "src/shared_motion.py",
                "scripts/shared_motion_data.py",
            )
        },
        "torch": str(torch.__version__),
        "device": str(device),
        "train_clips": len(train.clips),
        "validation_clips": len(val.clips),
        "train_sources": len({row["source"] for row in train.rows}),
        "validation_sources": len({row["source"] for row in val.rows}),
        "train_windows": len(train),
        "validation_windows": len(val),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    last = args.output / "last.pt"
    if last.exists() != args.resume:
        raise ValueError("Use --resume only with an existing last.pt, or choose a new output")
    index_rng = torch.Generator().manual_seed(cfg["seed"])
    noise_rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    model = Denoiser(cfg["width"], cfg["depth"]).to(device)
    diffusion = build_process(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    start, best, history = 0, float("inf"), []
    if args.resume:
        previous = torch.load(last, map_location="cpu", weights_only=True)
        recorded = {key: provenance.get(key) for key in previous["provenance"]}
        if previous["config"] != cfg or previous["provenance"] != recorded:
            raise ValueError("Resume requires the same configuration, provenance and device")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        index_rng.set_state(previous["index_rng"])
        noise_rng.set_state(previous["noise_rng"])
        mean, scale = previous["mean"], previous["scale"]
        start, best, history = previous["step"], previous["best"], previous["history"]
    else:
        mean, scale = statistics(train)
    mean, scale = mean.to(device), scale.to(device)
    (args.output / "run.json").write_text(
        json.dumps({"config": cfg, "provenance": provenance, "representation": REPRESENTATION}, indent=2) + "\n"
    )
    print(json.dumps(provenance), flush=True)
    tic, total, count = time.monotonic(), 0.0, 0
    elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for step in range(start + 1, cfg["steps"] + 1):
        model.train()
        clean = draw(train, cfg["batch_size"], index_rng, device)
        loss = diffusion.loss(model, (clean - mean) / scale, noise_rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item()
        count += 1
        if step % cfg["validate_every"] and step != cfg["steps"]:
            continue
        score = validation(model, diffusion, val, cfg, mean, scale, device)
        row = {"step": step, "train_mse": total / count, "val_mse": score,
               "elapsed_seconds": round(elapsed + time.monotonic() - tic, 2)}
        if device.type == "cuda":
            row["peak_cuda_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        history.append(row)
        total, count = 0.0, 0
        checkpoint = {
            "representation": REPRESENTATION, "config": cfg, "provenance": provenance,
            "step": step, "history": history, "best": min(best, score),
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "mean": mean.cpu(), "scale": scale.cpu(), "hand_means": torch.from_numpy(means),
            "index_rng": index_rng.get_state(), "noise_rng": noise_rng.get_state(),
        }
        if score < best:
            atomic_save(checkpoint, args.output / "best.pt")
            best = score
        atomic_save(checkpoint, last)
        with (args.output / "learning.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset", "asset", "config", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
