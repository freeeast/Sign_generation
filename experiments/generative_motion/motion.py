"""Dataset windows and SMPL-X conversion shared by training and evaluation."""
import csv
from pathlib import Path

import numpy as np
import torch

from scripts.shared_motion_data import ACTIVE, KEYS, WIDTHS
from src.shared_motion import decode, hand_mean, rotations, six, six_to_matrix

GROUPS = {"body": slice(0, 13), "left": slice(13, 28), "right": slice(28, 43)}
FEATURES = len(ACTIVE) * 6
REPRESENTATION = "shared-motion-active-43-rot6d-v1"


def split_rows(dataset):
    with (dataset / "sample.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = {split: [] for split in ("train", "val", "test")}
    source_split = {}
    clips = set()
    for row in rows:
        split, source, clip = (row[key] for key in ("split", "source", "clip"))
        if split not in groups or not source or not clip or clip in clips:
            raise ValueError(f"Invalid or duplicate clip: {clip}")
        if source_split.setdefault(source, split) != split:
            raise ValueError(f"Recording source leaks between splits: {source}")
        clips.add(clip)
        groups[split].append(row)
    return groups


def fit_rotations(dataset, row, means, fps):
    root = dataset.resolve()
    path = (root / row["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Motion path escapes dataset: {path}")
    required = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose")
    with np.load(path, allow_pickle=False) as source:
        if "fps" in source and float(source["fps"].item()) != fps:
            raise ValueError(f"Unexpected FPS in {path}")
        fit = {key: source[key] for key in KEYS if key in source}
    frames = int(row["frames"])
    if any(key not in fit for key in required):
        raise ValueError(f"Missing required SMPL-X field in {path}")
    if any(fit[key].shape != (frames, width) or not np.isfinite(fit[key]).all()
           for key, width in zip(KEYS, WIDTHS) if key in fit):
        raise ValueError(f"Invalid rotation arrays or frame count in {path}")
    return rotations(fit, means)


class Windows:
    def __init__(self, dataset, rows, means, frames, stride, fps):
        self.clips = []
        self.indices = []
        self.rows = []
        for row in rows:
            if int(row["frames"]) < frames:
                continue
            matrix = fit_rotations(dataset, row, means, fps)
            features = six(torch.from_numpy(matrix)).reshape(len(matrix), FEATURES)
            starts = list(range(0, len(matrix) - frames + 1, stride))
            if starts[-1] != len(matrix) - frames:
                starts.append(len(matrix) - frames)
            index = len(self.clips)
            self.clips.append(features)
            self.rows.append(row)
            self.indices.extend((index, start) for start in starts)
        self.frames = frames

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        clip, start = self.indices[index]
        return self.clips[clip][start:start + self.frames]


def matrix_from_features(features):
    vectors = features.reshape(*features.shape[:-1], len(ACTIVE), 6)
    a, b = vectors[..., :3], vectors[..., 3:]
    cross = torch.linalg.cross(a, b)
    if (a.norm(dim=-1) < 1e-6).any() or (cross.norm(dim=-1) < 1e-6).any():
        raise ValueError("Degenerate 6D rotation in generated motion")
    matrix = six_to_matrix(vectors)
    if not torch.isfinite(matrix).all():
        raise ValueError("Nonfinite rotation in generated motion")
    return matrix


def export_smplx(features, means, fps, output):
    matrix = matrix_from_features(features).cpu().numpy()
    frames = len(matrix)
    scaffold = {"body_pose": np.zeros((frames, 63), np.float32)}
    fit = decode(scaffold, matrix, means)
    fit.update(
        transl=np.zeros((frames, 3), np.float32),
        betas=np.zeros((1, 10), np.float32),
        expression=np.zeros((1, 10), np.float32),
        fps=np.array(fps),
    )
    np.savez_compressed(output, **fit)
