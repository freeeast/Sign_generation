"""COCO-WholeBody pose input. Coordinates use x-right / y-down normalized image space."""
from pathlib import Path
import pickle

import numpy as np

BODY = np.array([0, *range(3, 11)])
LEFT = np.arange(91, 112)
RIGHT = np.arange(112, 133)
FACE = np.array([*range(23, 40, 2), *range(83, 91), 53])
FINGERS = np.r_[92:112, 113:133]


def validate(xy, score):
    xy, score = np.asarray(xy, dtype=np.float32), np.asarray(score, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[1:] != (133, 2) or len(xy) == 0:
        raise ValueError(f"Expected keypoints (T,133,2), got {xy.shape}")
    if score.shape != xy.shape[:2]:
        raise ValueError(f"Expected scores {xy.shape[:2]}, got {score.shape}")
    if not np.isfinite(xy).all() or not np.isfinite(score).all():
        raise ValueError("Nonfinite coordinates/scores")
    return xy, score


def load_pose(path, *, trusted_pickle=False):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return validate(data["keypoints"], data["scores"])
    if path.suffix != ".pkl" or not trusted_pickle:
        raise ValueError("Use .npz, or use --trusted-pickle for verified upstream .pkl artifacts")
    with path.open("rb") as stream:
        # Python pickle can execute code. Use only verified upstream artifact bytes.
        data = pickle.load(stream)
    xy, score = np.asarray(data["keypoints"]), np.asarray(data["scores"])
    if xy.ndim == 4 and xy.shape[1] == 1 and score.ndim == 3 and score.shape[1] == 1:
        xy, score = xy[:, 0], score[:, 0]
    return validate(xy, score)
