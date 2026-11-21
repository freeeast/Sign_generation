"""Check the migrated archive, with optional GRU and external-asset checks."""

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
SHARED = REPO / "experiments/SharedMotion"
CHECKPOINT = SHARED / "model/gru_context16_h3.pt"


def archived_paths():
    yield from (
        "AGENTS.md",
        "HANDOFF_CONTEXT.md",
        "START_HERE_CN.md",
        "CODEX_START_PROMPT.md",
        "src/shared_motion.py",
        "scripts/shared_motion_feedback.py",
        "scripts/stage10_run.py",
        "configs/stage9.json",
        "experiments/SharedMotion/model/gru_context16_h3.pt",
        "experiments/SharedMotion/model/test_forecast.csv",
        "experiments/SharedMotion/model/test_per_clip.csv",
        "experiments/SharedMotion/analysis/paired_source_bootstrap.csv",
        "experiments/SharedMotion/data/sample.csv",
        "experiments/SharedMotion/feedback/integration.csv",
        "experiments/SharedMotion/visuals/Video4_feedback_integration.mp4",
    )
    config = json.loads((REPO / "configs/stage10.json").read_text(encoding="utf-8"))
    for source in config["development"]:
        yield f"experiments/Stage7/data/inputs/{source}.npz"
        yield f"experiments/Stage7/data/inputs/{source}.mp4"
        yield f"experiments/Stage7/data/paired/{source}.npz"
        yield f"experiments/Stage7/data/real_pose/{source}.npz"
        coarse = (
            "Stage9/new_source/stage8_controls" if source.startswith("DKO") else "Stage8"
        )
        yield f"experiments/{coarse}/{source}/B/initial_fit.npz"
        yield f"experiments/{coarse}/{source}/frames.json"
        for arm in ("G_reanchored_geometry", "I_matched_geometry", "H_reanchored_guarded", "B_stage9_joint"):
            yield f"experiments/Stage10/{source}/{arm}/final_fit.npz"


def smoke():
    sys.path.insert(0, str(REPO))
    import torch

    from src.shared_motion import ACTIVE, load_predictor

    torch.set_num_threads(1)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            "Restore GRU weights from the owner-supplied migration ZIP first. "
            "See CODEX_CLOUD_SETUP.md and scripts/restore_migration_assets.py."
        )
    predictor, metadata = load_predictor(CHECKPOINT)
    expected_metadata = (43, 16, 3, 165633)
    observed_metadata = (
        len(ACTIVE),
        metadata["context"],
        metadata["training_horizon"],
        sum(p.numel() for p in predictor.parameters()),
    )
    if observed_metadata != expected_metadata:
        raise RuntimeError(f"Unexpected predictor metadata: {observed_metadata}")

    history = torch.eye(3).expand(1, metadata["context"], len(ACTIVE), 3, 3).clone()
    velocity = torch.zeros(1, metadata["context"], len(ACTIVE), 3)
    with torch.inference_mode():
        rollout, deltas = predictor.rollout(history, velocity, 6)
    if tuple(rollout.shape) != (1, 6, 43, 3, 3) or tuple(deltas.shape) != (1, 6, 43, 3):
        raise RuntimeError("Unexpected rollout shape")
    if not torch.isfinite(rollout).all() or not torch.isfinite(deltas).all():
        raise RuntimeError("Non-finite values in GRU rollout")
    identity = torch.eye(3).expand_as(rollout)
    ortho_error = (rollout.transpose(-1, -2) @ rollout - identity).abs().max().item()
    det_error = (torch.linalg.det(rollout) - 1).abs().max().item()
    if max(ortho_error, det_error) > 1e-5:
        raise RuntimeError("Invalid rotation matrices in GRU rollout")
    print(f"GRU smoke PASS (synthetic history, {tuple(rollout.shape)}, torch {torch.__version__})")
    print(f"  max orthogonality error={ortho_error:.3g}; max determinant error={det_error:.3g}")


def assets(work):
    paths = (
        ("SMPL-X neutral mesh", work / "models/smplx/SMPLX_NEUTRAL.npz"),
        ("Uni-Sign source", work / "Uni-Sign/models.py"),
        ("Uni-Sign SLT checkpoint", work / "how2sign_pose_only_slt.pth"),
        ("RTMW ONNX (RGB only)", work / "rtm/pose/end2end.onnx"),
        ("mT5 config (translation only)", work / "mt5/config.json"),
        ("mT5 tokenizer (translation only)", work / "mt5/spiece.model"),
        ("mT5 tokenizer config (translation only)", work / "mt5/tokenizer_config.json"),
    )
    missing = []
    for label, path in paths:
        status = "found" if path.is_file() else "missing"
        print(f"  {status:7} {label}: {path}")
        if status == "missing":
            missing.append(label)
    detectors = sorted((work / "rtm/detector").rglob("end2end.onnx"))
    detector_found = len(detectors) == 1 and detectors[0].is_file()
    print(f"  {'found' if detector_found else 'check':7} RTMW person detector (RGB only): {detectors}")
    if not detector_found:
        missing.append("single RTMW person detector")
    if not shutil.which("ffmpeg"):
        print("  missing FFmpeg (RGB only) on PATH")
        missing.append("FFmpeg")
    print("  Presence only. See staged reproduction guides for hashes, model versions and EGL.")
    return bool(missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run synthetic CPU GRU inference")
    parser.add_argument("--data", action="store_true", help="also require historical motion/video assets")
    parser.add_argument("--assets", action="store_true", help="inventory external assets")
    parser.add_argument("--work-dir", type=Path, default=REPO / "work")
    options = parser.parse_args()
    paths = tuple(archived_paths())
    if not options.data:
        paths = tuple(path for path in paths if path in {
            "AGENTS.md", "HANDOFF_CONTEXT.md", "START_HERE_CN.md", "CODEX_START_PROMPT.md",
            "src/shared_motion.py", "scripts/shared_motion_feedback.py", "scripts/stage10_run.py",
            "configs/stage9.json",
            "experiments/SharedMotion/model/test_forecast.csv",
            "experiments/SharedMotion/model/test_per_clip.csv",
            "experiments/SharedMotion/analysis/paired_source_bootstrap.csv",
            "experiments/SharedMotion/data/sample.csv",
            "experiments/SharedMotion/feedback/integration.csv",
        })
    missing = [path for path in paths if not (REPO / path).is_file()]
    if missing:
        print("Missing archive files:", *missing, sep="\n  ", file=sys.stderr)
        return 1
    print(f"Archive layout PASS ({len(paths)} required paths)")
    if options.smoke:
        smoke()
    if options.assets and assets(options.work_dir.resolve()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
