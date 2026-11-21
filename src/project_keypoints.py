"""Project labeled 3D landmarks into normalized x-right/y-down image coordinates."""
import argparse
from pathlib import Path

import numpy as np

from .pose import BODY, validate


def orthographic(points, mapped, right, down, aspect, scale, center=None):
    points = np.asarray(points)
    mapped = np.asarray(mapped, dtype=bool)
    right, down = np.asarray(right, dtype=float), np.asarray(down, dtype=float)
    if points.ndim != 3 or points.shape[1:] != (133, 3) or not len(points):
        raise ValueError("Expected nonempty 3D points (T,133,3)")
    if mapped.shape != (133,) or not mapped[BODY].all() or not np.isfinite(points).all():
        raise ValueError("Need finite coordinates and all 9 mapped body landmarks")
    if not np.allclose([np.linalg.norm(right), np.linalg.norm(down), right @ down], [1, 1, 0]):
        raise ValueError("Right and down axes must be orthonormal")
    if not np.isfinite([aspect, scale]).all() or aspect <= 0 or scale <= 0:
        raise ValueError("Aspect and scale must be positive")
    plane = points @ np.stack((right, down), axis=1)
    if center is None:
        upper = plane[:, BODY].reshape(-1, 2)
        center = (upper.min(axis=0) + upper.max(axis=0)) / 2
    image = (plane - center) * [scale / aspect, scale] + 0.5
    image[:, ~mapped] = 0.0
    return image


def plot_frames(xy, score, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    upper_links = [(0, 3), (0, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
    for ax, frame in zip(axes, np.linspace(0, len(xy) - 1, 4, dtype=int)):
        points, visible = xy[frame], score[frame] > 0.3
        parts = [(BODY, "#25334b"), (np.arange(23, 91), "#758298")]
        for wrist, color in ((91, "#146f9f"), (112, "#b6501e")):
            parts.append((np.arange(wrist, wrist + 21), color))
            for finger in range(5):
                order = [wrist, *range(wrist + 1 + 4 * finger, wrist + 5 + 4 * finger)]
                for a, b in zip(order, order[1:]):
                    if visible[a] and visible[b]:
                        ax.plot(*points[[a, b]].T, lw=.8, c=color)
        for ids, color in parts:
            active = ids[visible[ids]]
            ax.scatter(*points[active].T, c=color, s=8)
        for a, b in upper_links:
            if visible[a] and visible[b]:
                ax.plot(*points[[a, b]].T, lw=1, c="#25334b")
        for idx, label in ((5, "L"), (6, "R")):
            if visible[idx]:
                ax.annotate(label, points[idx], xytext=(4, 4), textcoords="offset points")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_title(f"Frame {frame}")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NPZ keys: landmarks3d, mapped")
    parser.add_argument("output", type=Path)
    parser.add_argument("--right", nargs=3, type=float, required=True)
    parser.add_argument("--down", nargs=3, type=float, required=True)
    parser.add_argument("--aspect", type=float, default=1., help="Image width / height")
    parser.add_argument("--scale", type=float, default=1., help="Image heights per model unit")
    parser.add_argument("--center", nargs=2, type=float, help="Optional center in projected model units")
    parser.add_argument("--confidence", type=float, required=True, help="Assumption for mapped landmarks")
    parser.add_argument("--score-track", type=Path, help="NPZ with frame-aligned scores, shape (T,133)")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        mapped = data["mapped"].astype(bool)
        xy = orthographic(data["landmarks3d"], mapped, args.right, args.down,
                          args.aspect, args.scale, args.center)
    score = np.broadcast_to(mapped * args.confidence, xy.shape[:2])
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be in [0,1]")
    if args.score_track:
        with np.load(args.score_track, allow_pickle=False) as track:
            measured, expected_shape = track["scores"], xy.shape[:2]
            if measured.shape != expected_shape:
                raise ValueError(f"Score track must have aligned shape {expected_shape}")
            score = measured * mapped
    xy, score = validate(xy, score)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keypoints=xy, scores=score)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_frames(xy, score, args.plot)


if __name__ == "__main__":
    main()
