"""Run source-held-out SMPL-X closed loops with the frozen Stage 5 scaling."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.stage5_closed_loop import ARMS, Pilot
from src.run_real_validation import load_manifest, observations
from src.stage4_features import FrozenFeedback
from src.stage5_motion import MotionSpace
from src.unisign import UniSign
from src.virtual_pose import load_fit


class DirectPilot(Pilot):
    arms = {k: v for k, v in ARMS.items() if k != "C_nll"}
    arms.update(C_temporal_guard=(True, "hand_temporal"),
                C_temporal_staged=(True, "hand_temporal"))

    def __init__(self, args, cfg, row, expert, temp):
        self.args, self.cfg, self.row, self.expert, self.temp = args, cfg, row, expert, temp
        self.clip = row["clip_id"]
        self.fit = load_fit(args.motions / f"{self.clip}.npz")
        self.space = MotionSpace(self.fit, args.asset, cfg)
        self.zero = np.zeros(len(cfg["corruption_coeff_deg"]))
        self.cache = {}
        real, scores, self.wh = observations(args.output / "data/real_pose", row)
        self.real = np.where(scores[..., None] > cfg["rtmw_threshold"], real, np.nan)
        self.ref = expert.extract(real[::2], scores[::2])
        self.ones = np.ones(scores[::2].shape, np.float32)
        with np.load(args.output / "data/paired" / f"{self.clip}.npz") as data:
            self.mp, self.original3d = data["mp"], data["original3d"]
            self.camera, self.original2d = data["camera"], data["original2d"]
        self.metric_original = self.measure(self.original2d, self.original3d)

    def measure(self, xy, points):
        # Stage 6 uses encoder signals exclusively. Skip the Stage 5 decoder pass.
        return super().measure(xy, points, target_sentence=False)

    def loss(self, metrics, base, coef, arm, scales, *, step=0, axis=None):
        if metrics["constraint_violations"]:
            return float("inf")
        if arm in {"C_temporal_guard", "C_temporal_staged"}:
            if metrics["rtmw_even_wrists"] > base["rtmw_even_wrists"] + self.cfg["guard_wrist_slack_px"]:
                return float("inf")
            if arm == "C_temporal_staged":
                if np.max(np.abs(coef)) > self.cfg["strict_trust_coeff_deg"]:
                    return float("inf")
                if step <= self.cfg["early_checkpoint_steps"] or axis is None or axis >= 2:
                    arm = "B_geometry"
            if arm != "B_geometry":
                arm = "C_temporal"
        return super().loss(metrics, base, coef, arm, scales)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "checkpoint", "mt5", "asset", "motions", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clips", nargs="*")
    args = parser.parse_args()
    cfg = json.loads(Path("configs/stage5.json").read_text())
    cfg.update(json.loads(Path("configs/stage6.json").read_text()))
    scales = json.loads(Path("experiments/Stage5/calibration_scales.json").read_text())
    rows = load_manifest(args.output / "data/manifest.json")
    expert = FrozenFeedback(UniSign(args.repo, args.checkpoint, args.mt5, args.device, args.threads))
    with tempfile.TemporaryDirectory() as scratch:
        for row in rows:
            if not args.clips or row["clip_id"] in args.clips:
                pilot = DirectPilot(args, cfg, row, expert, Path(scratch)/"motion.npz")
                pilot.run(args.output/"direct", scales, "new_source")


if __name__ == "__main__":
    main()
