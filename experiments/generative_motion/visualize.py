"""Export official SMPL-X FK joints and fixed-view MP4 previews."""
import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.smplx_to_pose import BODY_NAMES, forward


def skeletons():
    body = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (12, 14), (13, 15), (14, 16), (5, 0), (6, 0)]
    hands = []
    for root in (91, 112):
        links = []
        for finger in range(5):
            chain = [root, *range(root + 1 + 4 * finger, root + 5 + 4 * finger)]
            links.extend(zip(chain[:-1], chain[1:]))
        hands.append(links)
    return [body, *hands]


def video(path, tracks, fps, label):
    views = (("Front", (0, 1), slice(0, len(BODY_NAMES))),
             ("Side", (2, 1), slice(0, len(BODY_NAMES))),
             ("Left hand, fixed crop", (0, 1), slice(91, 112)),
             ("Right hand, fixed crop", (0, 1), slice(112, 133)))
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=100)
    fig.subplots_adjust(wspace=.15, hspace=.22)
    groups = skeletons()
    artists = []
    for num, (ax, (title, dims, limits)) in enumerate(zip(axes.flat, views)):
        xy = tracks[:, limits][:, :, list(dims)].copy()
        low, high = xy.min(axis=(0, 1)), xy.max(axis=(0, 1))
        mid = (low + high) / 2
        radius = max(.19, float((high - low).max()) * .56)
        if num < 2:
            mid = (0, .30)
            radius = .88
        ax.set(xlim=(mid[0] - radius, mid[0] + radius), ylim=(mid[1] - radius, mid[1] + radius),
               aspect="equal", title=title)
        ax.axis("off")
        groups_to_show = (0, 1, 2) if num < 2 else (num - 1,)
        lines = []
        for group in groups_to_show:
            for edge in groups[group]:
                line, = ax.plot([], [], color=("#364758", "#dc5937", "#167e85")[group], lw=2)
                lines.append((edge, line))
        artists.append((dims, lines))
    stamp = fig.suptitle("")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgba", "-s", "800x800", "-r", str(fps), "-i", "-", "-an",
               "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", str(path)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            for i, frame in enumerate(tracks):
                stamp.set_text(f"{label}   |   {i / fps:.2f} s")
                for (x, y), lines in artists:
                    for (a, b), line in lines:
                        line.set_data(frame[[a, b], x], frame[[a, b], y])
                fig.canvas.draw()
                process.stdin.write(fig.canvas.buffer_rgba())
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg encoding failed")
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True, nargs="+")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", help="caption prefix for all input videos")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for motion in args.motion:
        with np.load(motion, allow_pickle=False) as data:
            fps = float(data["fps"])
        joints, mapped, mesh, _ = forward(
            motion, args.model_dir, "neutral", "model", 24, "cpu", True, return_mesh=True
        )
        if len(joints) < 2 or not np.isfinite(joints).all() or not np.isfinite(mesh).all():
            raise ValueError(f"Nonfinite forward kinematics: {motion}")
        np.savez_compressed(args.output / f"{motion.stem}_joints.npz", landmarks3d=joints, mapped=mapped, fps=fps)
        video(args.output / f"{motion.stem}.mp4", joints, fps, f"{args.label or ''} {motion.stem}".strip())
        metrics = {"frames": len(mesh), "vertices": mesh.shape[1]}
        for name, wrist in (("left", 91), ("right", 112)):
            points = joints[:, [wrist + 4 * digit for digit in range(1, 6)]]
            relative = points - joints[:, wrist, None]
            speed = np.linalg.norm(np.diff(relative, axis=0), axis=-1) * fps
            aperture = np.linalg.norm(points[:, 0] - points[:, 4], axis=-1)
            metrics[name] = {
                "wrist_relative_tip_speed_median_mm_s": float(np.median(speed) * 1000),
                "thumb_little_aperture_range_mm": float(np.ptp(aperture) * 1000),
            }
        (args.output / f"{motion.stem}_fk.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
