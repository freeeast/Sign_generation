"""Convert canonical axis-angle SMPL-X arrays to ordered 3D COCO-WholeBody landmarks."""
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
CONTOUR = ([f"right_contour_{i}" for i in range(1, 9)] + ["contour_middle"] +
           [f"left_contour_{i}" for i in range(8, 0, -1)])
# SMPL-X static face landmarks follow the 68-point ordering after its first 17 contour points.
FACE = [
    *CONTOUR, *[f"right_eye_brow{i}" for i in range(1, 6)],
    *[f"left_eye_brow{i}" for i in range(5, 0, -1)], *[f"nose{i}" for i in range(1, 5)],
    "right_nose_2", "right_nose_1", "nose_middle", "left_nose_1", "left_nose_2",
    *[f"right_eye{i}" for i in range(1, 7)], *[f"left_eye{i}" for i in (4, 3, 2, 1, 6, 5)],
    "right_mouth_1", "right_mouth_2", "right_mouth_3", "mouth_top", "left_mouth_3",
    "left_mouth_2", "left_mouth_1", "left_mouth_5", "left_mouth_4", "mouth_bottom",
    "right_mouth_4", "right_mouth_5", "right_lip_1", "right_lip_2", "lip_top",
    "left_lip_2", "left_lip_1", "left_lip_3", "lip_bottom", "right_lip_3",
]


def hand_names(side):
    names = [f"{side}_wrist"]
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        names.extend(f"{side}_{finger}{suffix}" for suffix in ("1", "2", "3", ""))
    return names


COCO_TO_SMPLX = BODY_NAMES + FACE + hand_names("left") + hand_names("right")


def map_landmarks(joints, joint_names, mask_face=False):
    if joints.ndim != 3 or joints.shape[1:] != (len(joint_names), 3):
        raise ValueError(f"Unexpected SMPL-X joint count or shape: {joints.shape}")
    lookup = {name: position for position, name in enumerate(joint_names)}
    indices = [lookup[name] for name in COCO_TO_SMPLX]
    landmarks = joints[:, indices].copy()
    mapped = np.ones((133,), dtype=bool)
    if mask_face:
        mapped[23:91] = False
        landmarks[:, 23:91] = 0.
    return landmarks, mapped


@lru_cache(maxsize=2)
def load_model(model_dir, gender, hand_mean, device, widths):
    import smplx
    return smplx.create(str(model_dir), model_type="smplx", gender=gender, ext="npz",
                        use_pca=False, flat_hand_mean=(hand_mean == "flat"),
                        use_face_contour=True, **dict(widths)).eval().to(device)


def forward(input_path, model_dir, gender, hand_mean, batch_size, device, mask_face):
    import torch
    from smplx.joint_names import JOINT_NAMES

    required = {"global_orient": 3, "body_pose": 63, "left_hand_pose": 45,
                "right_hand_pose": 45}
    optional = {key: 3 for key in ("transl", "jaw_pose", "leye_pose", "reye_pose")}
    with np.load(input_path, allow_pickle=False) as motion:
        frames = len(motion["global_orient"])
        if frames < 1:
            raise ValueError("Empty motion clip")
        values = {}
        for key, width in (required | optional).items():
            if key in required or key in motion:
                values[key] = motion[key]
                if values[key].shape != (frames, width):
                    raise ValueError(f"{key} must have axis-angle shape {(frames, width)}")
        for key in ("betas", "expression"):
            if key in motion:
                data = motion[key]
                if data.ndim == 1:
                    data = data[None, :]
                if data.ndim != 2 or data.shape[0] not in (1, frames) or data.shape[1] < 1:
                    raise ValueError(f"{key} must have shape (K,), (1,K), or (T,K)")
                values[key] = np.broadcast_to(data, (frames, data.shape[1])).copy()
    if any(not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite SMPL-X parameters")
    widths = {key: values[array].shape[1] for key, array in
              [("num_betas", "betas"), ("num_expression_coeffs", "expression")] if array in values}
    model = load_model(Path(model_dir).resolve(), gender, hand_mean, device, tuple(widths.items()))
    # Supply per-frame neutral values where the fit omitted optional parameters.
    for key, width in (optional | {"betas": model.num_betas,
                                    "expression": model.num_expression_coeffs}).items():
        values.setdefault(key, np.zeros((frames, width), np.float32))
    chunks = []
    with torch.inference_mode():
        for start in range(0, frames, batch_size):
            params = {key: torch.as_tensor(value[start:start + batch_size], dtype=torch.float32, device=device)
                      for key, value in values.items()}
            chunks.append(model(**params, return_verts=False).joints.cpu().numpy())
    return map_landmarks(np.concatenate(chunks), JOINT_NAMES, mask_face=mask_face)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--gender", choices=("neutral", "male", "female"), required=True)
    parser.add_argument("--hand-mean", choices=("flat", "model"), required=True)
    parser.add_argument("--mask-face", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    landmarks, mapped = forward(args.motion, args.model_dir, args.gender, args.hand_mean,
                                args.batch_size, args.device, args.mask_face)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, landmarks3d=landmarks, mapped=mapped)


if __name__ == "__main__":
    main()
