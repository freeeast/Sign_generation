"""Produce Stage 5 controlled contrasts, optimization plots, and motion videos."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.stage4_review_media import render_pair
from src.evaluate_feedback import write_csv

METRICS = (
    "rtmw_even_combined", "rtmw_odd_combined", "mp_odd_combined",
    "mp_odd_tips", "mp_odd_wrists", "mp_odd_handshape", "hand_temporal",
    "hand_spatial", "global_encoder", "nll", "fit_distance_3d_mm",
    "hand_accel_p95_mm_frame2", "bone_length_drift_p95_m",
)
LABELS = {
    "B_geometry":"B geometry", "C_temporal":"C hand temporal",
    "D_temporal_only":"D expert only", "C_spatial":"C hand spatial",
    "C_global":"C global", "C_nll":"C NLL",
}


def load(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def contrasts(root, rows):
    summary = []
    for row in rows:
        clip = row["clip"]
        if row["method"] not in LABELS:
            continue
        bank = {r["method"]: r for r in rows if r["clip"]==clip}
        for label in ("I_perturbed", "B_geometry", "A_original"):
            baseline = bank[label]
            summary.append(dict(clip=clip, split=row["split"], method=row["method"], reference=label,
                                **{f"delta_{k}": float(row[k])-float(baseline[k]) for k in METRICS}))
    write_csv(root/"contrasts.csv",summary)
    diagnostic = []
    for row in rows:
        clip, method = row["clip"], row["method"]
        if method not in LABELS:
            continue
        trace = load(root/clip/f"{method}_trajectory.csv")
        best = min(trace,key=lambda t:float(t["mp_odd_combined"]))
        diagnostic.append(dict(clip=clip,split=row["split"],method=method,
                               retrospective_best_mp_step=int(best["step"]),
                               best_mp_odd_px=float(best["mp_odd_combined"]),
                               final_minus_best_mp_px=float(trace[-1]["mp_odd_combined"])-float(best["mp_odd_combined"]),
                               final_minus_start_hand_temporal=float(trace[-1]["hand_temporal"])-float(trace[0]["hand_temporal"])))
    write_csv(root/"trajectory_diagnostics.csv",diagnostic)


def draw(directory, clip, render):
    rows = load(directory/"comparison.csv")
    keys = ("hand_temporal", "rtmw_even_combined", "mp_odd_combined")
    labels = ("Real–virtual hand feature distance", "Optimization geometry (px)", "Odd-frame MediaPipe audit (px)")
    colors = ("#355d86", "#9c421f", "#646b68", "#766ba8", "#5c8f52", "#b56892")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for method, color in zip(LABELS, colors):
        trajectory = load(directory/f"{method}_trajectory.csv")
        n = [int(r["step"]) for r in trajectory]
        for ax, key in zip(axes, keys):
            ax.plot(n,[float(r[key]) for r in trajectory],label=LABELS[method],color=color,marker="." if method=="C_temporal" else None)
    orig = next(row for row in rows if row["method"]=="A_original")
    shared = next(row for row in rows if row["method"]=="I_perturbed")
    for ax, key, title in zip(axes, keys, labels):
        ax.axhline(float(orig[key]),ls="--",lw=.8,color=".4",label="A original")
        ax.axhline(float(shared[key]),ls=":",lw=.7,color=".55",label="I perturbation")
        ax.set(title=title,xlabel="Update step")
        ax.spines[["top","right"]].set_visible(False)
    axes[-1].legend(fontsize=7)
    fig.suptitle(clip)
    fig.savefig(directory/"optimization_curves.png",dpi=155)
    plt.close(fig)
    if render:
        cases = [("I_perturbed","C_temporal"),("B_geometry","C_temporal"),("I_perturbed","D_temporal_only")]
        with np.load(directory/"final_motion_tracks.npz") as f:
            for left,right in cases:
                render_pair(directory/f"{left}_vs_{right}.mp4",
                            [f[k+"__xy"] for k in (left,right)],
                            [f[k+"__points"] for k in (left,right)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--no-video",action="store_true")
    args=parser.parse_args()
    rows=load(args.root/"comparison.csv")
    contrasts(args.root,rows)
    for clip in sorted({r["clip"] for r in rows}):
        draw(args.root/clip,clip,not args.no_video)


if __name__=="__main__":
    main()
