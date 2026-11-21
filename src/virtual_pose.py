"""Use the existing SMPL-X FK and derive camera/observation controls for real fits."""
import hashlib
from pathlib import Path

import numpy as np

from .pose import BODY, FACE, FINGERS, LEFT, RIGHT, validate
from .project_keypoints import orthographic
from .smplx_to_pose import forward

CALIBRATION_POINTS = np.array([0, 3, 4, 5, 6])
TIPS = np.r_[95:112:4, 116:133:4]
WIDE = {"body": BODY, "wrists": [9, 10], "fingertips": TIPS, "fingers": FINGERS, "face": FACE}


def load_fit(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def pinhole(points, cam_trans):
    """OSX's virtual crop coordinates with x right and y down; no in-image calibration."""
    camera = points + cam_trans[:, None]
    if (camera[..., 2] <= 0).any():
        raise ValueError("OSX hypothesis yielded points behind camera")
    return camera[..., :2] / camera[..., 2, None] * [5000 / 192, 5000 / 256] + .5


def calibrate(xy, observed, confidence):
    """Fit a constant per-axis affine on even frames and only upper anchors."""
    frames = np.arange(0, len(xy), 2)
    virtual = xy[frames][:, CALIBRATION_POINTS]
    target = observed[frames][:, CALIBRATION_POINTS]
    mask = confidence[frames][:, CALIBRATION_POINTS] > .3
    if mask.sum() < 6:
        raise ValueError("Insufficient upper landmarks for camera diagnostic")
    transform = []
    for x, y in zip(virtual[mask].T, target[mask].T):
        coef = np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
        if not np.isfinite(coef).all() or coef[0] <= 0:
            raise ValueError("Calibration reversed an image axis or had invalid scale")
        transform.append(coef)
    return np.array(transform)


def apply_affine(xy, matrix):
    return np.einsum("...ij,...kj->...ki", matrix[..., :2], xy) + matrix[..., 2][..., None, :]


def virtual_inputs(motion_path, model_dir, real, real_score, wh, hand_mean, device,
                   framewise_crop=None):
    motion = load_fit(motion_path)
    points, mapped = forward(motion_path, model_dir, "neutral", hand_mean, 64, device, False)
    if len(points) != len(real):
        raise ValueError("Framewise alignment requires equal counts or an explicitly verified index map")
    if not np.array_equal(points[:, [9, 10]], points[:, [91, 112]]):
        raise ValueError("Body and hand wrist aliases should use the same anatomical joints")

    in_crop = pinhole(points, motion["cam_trans"])
    if framewise_crop is None:
        factors = calibrate(in_crop, real, real_score)
        camera = np.array([[factors[0, 0], 0, factors[0, 1]], [0, factors[1, 0], factors[1, 1]]])
        camera_type = "proxy_constant_affine_fit_on_even_upper_anchors"
    else:
        camera = np.asarray(framewise_crop)
        if camera.shape not in {(2, 3), (len(points), 2, 3)} or not np.isfinite(camera).all():
            raise ValueError("Crop affine must be (2,3) or (T,2,3) in normalized coordinates")
        camera_type = "supplied_fit_camera_affine"

    primary_xy = apply_affine(in_crop, camera)
    constant = np.broadcast_to(mapped.astype(np.float32), real_score.shape)
    borrowed = real_score * mapped
    face_mask = constant.copy()
    face_mask[:, 23:91] = 0

    shoulder_span = np.median(np.linalg.norm(points[:, 5] - points[:, 6], axis=1))
    if shoulder_span <= 0:
        raise ValueError("Degenerate shoulder width")
    plane = orthographic(points, mapped, [1, 0, 0], [0, 1, 0], wh[0] / wh[1],
                         .25 / shoulder_span)
    square_normalized = primary_xy.copy()
    square_normalized[..., 0] *= wh[0] / wh[1]

    variants = {"proxy_matched_constant1": (primary_xy, constant), "osx_crop_constant1": (in_crop, constant),
                "ortho_constant1": (plane, constant), "face_mask_only": (primary_xy, face_mask),
                "borrow_rtmscores_only": (primary_xy, borrowed), "constant0p7_only": (primary_xy, constant * .7),
                "pixel_aspect_normalization": (square_normalized, constant)}
    # Estimate separate detector hand-wrist vs body-wrist offsets on even frames.
    # Translate each hand as a whole so the hand-relative coordinates stay fixed.
    # This checks anchor sensitivity in geometry; the expected Uni-Sign contrast is near zero.
    shifts = primary_xy.copy()
    for hand, body_wrist in ((LEFT, 9), (RIGHT, 10)):
        visible = (real_score[::2, [hand[0], body_wrist]] > .3).all(axis=1)
        if visible.any():
            shift = np.median(real[::2][visible, hand[0]] - real[::2][visible, body_wrist], axis=0)
            shifts[:, hand] += shift
    variants["hand_anchor_only"] = (shifts, constant)
    variants = {name: validate(*value) for name, value in variants.items()}
    quality = geometry_metrics(points, primary_xy, real, real_score, wh)
    quality.update(camera_type=camera_type, camera=np.asarray(camera).tolist(),
                   shoulder_left_on_image_right_fraction=float((primary_xy[:, 5, 0] > primary_xy[:, 6, 0]).mean()),
                   observed_left_on_image_right_fraction=float((real[:, 5, 0] > real[:, 6, 0]).mean()),
                   positive_crop_depth_min=float((points[..., 2] + motion["cam_trans"][:, None, 2]).min()),
                   primary_outside_image_fraction=float(((primary_xy < 0) | (primary_xy > 1)).any(axis=-1).mean()))
    return points, variants, camera, quality


def geometry_metrics(points, xy, real, confidence, wh):
    frames = np.arange(1, len(xy), 2)
    delta = np.linalg.norm((xy - real) * np.asarray(wh), axis=-1)
    distances = {}
    for name, ids in WIDE.items():
        values = delta[frames][:, ids][confidence[frames][:, ids] > .3]
        distances[name + "_holdout_px"] = float(np.median(values)) if len(values) else None
    distances["hand_acceleration_m_per_frame2_median"] = float(np.median(
        np.linalg.norm(np.diff(points[:, FINGERS], n=2, axis=0), axis=-1))) if len(points) > 2 else None
    # A trajectory proxy, not a physiological joint-limit or contact check.
    return distances


def model_sha(model_dir):
    path = Path(model_dir)
    if path.is_dir():
        path = path / "smplx/SMPLX_NEUTRAL.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Need an authorized neutral SMPL-X model: {path}")
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()
