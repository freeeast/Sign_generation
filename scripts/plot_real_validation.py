"""Render the real input checks and the same-source sentence-ranking pilot."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pose import load_pose
from src.project_keypoints import plot_frames


def render(poses, motion, input_dir, output_dir, example):
    xy, score = load_pose(poses / (example + ".npz"))
    plot_frames(xy, score, input_dir / "real_rtmpose_keyframes.png")
    with np.load(motion, allow_pickle=False) as data:
        fields = dict(data)
    t = np.arange(len(xy)) / 24.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    axes[0].plot(t, fields["cam_trans"][:, 2], color="#22677d")
    axes[0].set(title="Stored camera depth", ylabel="Translation z (model units)")
    for side, color in (("left", "#147b9b"), ("right", "#b95d32")):
        fingers = fields[side + "_hand_pose"].reshape(-1, 15, 3)
        axes[1].plot(t, np.median(np.linalg.norm(fingers, axis=-1), axis=1), label=side, color=color)
        wrist = fields["body_pose"].reshape(-1, 21, 3)[:, 19 if side == "left" else 20]
        axes[2].plot(t, np.linalg.norm(wrist, axis=-1), label=side, color=color)
    axes[1].set(title="Hand articulation parameters", ylabel="Median joint angle (rad)")
    axes[2].set(title="Wrist orientation parameters", ylabel="Axis-angle norm (rad)")
    for ax in axes:
        ax.set_xlabel("Time (s), assuming 24 fps")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False)
    fig.savefig(input_dir / "uploaded_fit_parameter_tracks.png", dpi=170)
    plt.close(fig)

    with (output_dir / "real_only_nll_matrix.csv").open(newline="") as source:
        records = list(csv.DictReader(source))
    same = [r for r in records if r["pose_clip"].startswith("_2FBDaOPYig_") and r["same_video"] == "True"]
    ids = list(dict.fromkeys(r["pose_clip"] for r in same))
    matrix = np.array([[next(float(r["nll"]) for r in same if r["pose_clip"] == video and
                             r["reference_clip"] == ref) for ref in ids] for video in ids])
    margin = matrix - matrix.diagonal()[None, :]
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    bound = np.abs(margin).max()
    heat = ax.imshow(margin, vmin=-bound, vmax=bound, cmap="RdBu")
    labels = [i.removeprefix("_2FBDaOPYig_").split("-")[0] for i in ids]
    ax.set(xticks=range(len(ids)), xticklabels=labels, yticks=range(len(ids)), yticklabels=labels,
           xlabel="Reference sentence suffix", ylabel="Real pose clip suffix",
           title="Same recording: NLL versus each matched pose")
    for i in range(len(ids)):
        for j in range(len(ids)):
            ax.text(j, i, f"{margin[i, j]:+.3f}", ha="center", va="center",
                    color="white" if abs(margin[i, j]) > bound * .70 else "#172230")
    fig.colorbar(heat, ax=ax, label="Δ token NLL (nats)")
    fig.savefig(output_dir / "real_only_same_recording_margins.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("poses", "motion", "e1", "e3"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--example", default="_2FBDaOPYig_1-3-rgb_front")
    args = parser.parse_args()
    render(args.poses, args.motion, args.e1, args.e3, args.example)
