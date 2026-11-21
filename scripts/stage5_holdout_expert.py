"""Audit final saved endpoints on odd frames never supplied to the optimizer."""
import argparse
from pathlib import Path

import numpy as np

from src.evaluate_feedback import write_csv
from src.run_real_validation import load_manifest, observations
from src.stage4_features import FrozenFeedback, distances
from src.unisign import UniSign

ENDPOINTS = ("A_original", "I_perturbed", "B_geometry", "C_temporal", "D_temporal_only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root","repo","checkpoint","mt5"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    args = parser.parse_args()
    manifest = {r["clip_id"]:r for r in load_manifest(Path("configs/validation_pairs.json"))}
    expert = FrozenFeedback(UniSign(args.repo,args.checkpoint,args.mt5,args.device,4))
    records = []
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir() or not (directory/"final_motion_tracks.npz").exists():
            continue
        clip = directory.name
        real, scores, _ = observations(Path("experiments/E3/real_pose"),manifest[clip])
        reference = expert.extract(real[1::2],scores[1::2])
        with np.load(directory/"final_motion_tracks.npz",allow_pickle=False) as data:
            for method in ENDPOINTS:
                virtual = expert.extract(data[method+"__xy"][1::2],np.ones_like(scores[1::2]))
                records.append(dict(clip=clip,method=method,frames=len(real[1::2]),**distances(virtual,reference)))
    write_csv(args.root/"odd_frame_expert_audit.csv",records)


if __name__=="__main__":
    main()
