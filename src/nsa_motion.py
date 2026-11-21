"""Validate and convert a Neural Sign Actors frame ZIP to canonical SMPL-X arrays."""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import pickle
import re
import zipfile

import numpy as np

FIELDS = {
    "smplx_root_pose": ("global_orient", 3),
    "smplx_body_pose": ("body_pose", 63),
    "smplx_lhand_pose": ("left_hand_pose", 45),
    "smplx_rhand_pose": ("right_hand_pose", 45),
    "smplx_jaw_pose": ("jaw_pose", 3),
    "smplx_shape": ("betas", 10),
    "smplx_expr": ("expression", 10),
    "cam_trans": ("cam_trans", 3),
}


class ArrayUnpickler(pickle.Unpickler):
    """Read the producer's NumPy array records with a restricted global lookup."""
    def find_class(self, module, name):
        if (module, name) in {("numpy", "ndarray"), ("numpy", "dtype")}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name == "_reconstruct":
            return importlib.import_module(module)._reconstruct
        raise pickle.UnpicklingError(f"Unsupported global: {module}.{name}")


def convert(source, clip, output):
    pattern = re.compile(rf"{re.escape(clip)}_(\d+)_3D\.pkl")
    frames = {}
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if "__MACOSX" in path.parts or not (match := pattern.fullmatch(path.name)):
                continue
            index = int(match[1])
            if index in frames or item.file_size > 64 * 1024:
                raise ValueError(f"Duplicate or oversized frame {index}")
            record = ArrayUnpickler(io.BytesIO(archive.read(item))).load()
            if not isinstance(record, dict) or not FIELDS.keys() <= record.keys():
                raise ValueError(f"{clip}: missing fields in frame {index}")
            values = {}
            for original, (key, width) in FIELDS.items():
                data = np.asarray(record[original])
                if data.shape not in {(width,), (1, width)} or data.dtype.kind != "f" or not np.isfinite(data).all():
                    raise ValueError(f"{clip}: malformed {original} at frame {index}")
                values[key] = data.reshape(width).astype(np.float32)
            frames[index] = values
    if not frames or sorted(frames) != list(range(len(frames))):
        raise ValueError(f"{clip}: frame indices must be contiguous and start at zero")
    arrays = {key: np.stack([frames[i][key] for i in range(len(frames))]) for key in frames[0]}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary = {
        "clip_id": clip, "frame_count": len(frames), "archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fields": {k: {"shape": list(v.shape), "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist(),
                       "adjacent_step_norm_p95": float(np.percentile(np.linalg.norm(np.diff(v, axis=0), axis=-1), 95))
                       if len(v) > 1 else None} for k, v in arrays.items()},
        "missing": ["eye rotations", "focal", "principal point", "person crop", "timestamps", "FPS", "gender"],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("clip")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.clip, args.output)))
