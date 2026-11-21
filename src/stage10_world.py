"""Fixed virtual camera and selective body-coordinate stabilization."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform
import torch

from .stage9_hand import SparseFK, tensor
from .stage9_timing import rotvec_safe, rotation_matrices, p95
from .virtual_pose import apply_affine, pinhole

LOWER = np.array([1, 2, 4, 5, 7, 8, 10, 11])
GROUPS = {"pelvis": [0], "hip": [1,2], "knee": [4,5], "ankle": [7,8], "foot": [10,11], "wrist": [20,21]}
PIXELS = np.array([1280, 720])


def fixed_world(fit, camera_translation=None):
    """Gauge change that preserves the original camera-space projection."""
    old = fit["cam_trans"].astype(np.float64)
    offset = np.median(old, axis=0) if camera_translation is None else np.asarray(camera_translation)
    return dict(fit, cam_trans=np.broadcast_to(offset, old.shape).astype(np.float32).copy(),
                transl=(fit.get("transl", np.zeros_like(old))+old-offset).astype(np.float32))


def regularized(track, confidence, smoothing, anchoring):
    n = len(track)
    d = np.diff(np.eye(n), n=2, axis=0)
    w = 0.12 + np.clip(confidence, 0, 1)
    soft = anchoring * (1.10 - np.clip(confidence, 0, 1))
    anchor = np.median(track, axis=0)
    base = solve(np.diag(w+soft)+smoothing*d.T@d, w[:,None]*track+soft[:,None]*anchor,
                 assume_a="pos")
    # Huber reweighting reduces influence of short jumps, retaining sustained displacement.
    residual = np.linalg.norm(track-base, axis=-1)
    scale = max(1.4826*np.median(residual), 1e-5)
    w *= np.minimum(1., 2.0*scale/np.maximum(residual, 1e-9))
    return solve(np.diag(w+soft)+smoothing*d.T@d, w[:,None]*track+soft[:,None]*anchor,
                 assume_a="pos").astype(track.dtype)


def stabilized_body(fit, fit_indices, real_confidence, cfg):
    """Seated option. Use only optimization-frame confidences for body anchoring."""
    n = len(fit["body_pose"])
    confidence = np.clip(real_confidence,0,1)
    series = lambda ids: np.interp(np.arange(n),fit_indices,confidence[:,ids].mean(1))
    out = dict(fit)
    out["transl"] = regularized(fit["transl"], series([11,12]),
                               cfg["root_smoothness"], cfg["root_soft_anchor"])
    original = fit["body_pose"].reshape(n,21,3)
    body = original.copy()
    for label, source in [(0,[5,6,11,12]),*(zip(LOWER,[ [11],[12],[13],[14],[15],[16],[18,19,20],[17,21,22] ]))]:
        current = fit["global_orient"] if not label else original[:,label-1]
        matrices = rotation_matrices(current)
        center = Rotation.from_matrix(matrices).mean().as_matrix()
        tangent = rotvec_safe(center.T@matrices)
        if not label:
            filtered = regularized(tangent, series(source), 20, .10)
        else:
            filtered = regularized(tangent, series(source),cfg["leg_smoothness"],cfg["leg_soft_anchor"])
        result = rotvec_safe(center@rotation_matrices(filtered)).astype(current.dtype)
        if not label:
            out["global_orient"] = result
        else:
            body[:,label-1] = result
    out["body_pose"] = body.reshape(n,63)
    return out


def body_joints(fit, asset, aff, xy):
    f = SparseFK(fit, asset, aff, xy)
    with torch.no_grad():
        joints,_ = batch_rigid_transform(f.base_rot, f.rest, f.model.parents)
    return (joints + f.transl[:,None]).numpy()


def body_metrics(fit, joints, hands, projected, camera):
    # Use camera-aligned geometry for gauge-comparable measurements across legacy and fixed forms.
    c = fit["cam_trans"]
    effective_root = fit.get("transl",np.zeros_like(c))+c
    effective_joints = joints+c[:,None]
    xy = apply_affine(pinhole(joints,c), camera)*PIXELS
    row = {}
    for name, ids in GROUPS.items():
        a = effective_joints[:,ids]
        row[name+"_excursion_p95_mm"] = p95(np.linalg.norm(a-np.median(a,axis=0),axis=-1))*1000
        row[name+"_jerk_p95_mm_frame3"] = p95(np.linalg.norm(np.diff(a,n=3,axis=0),axis=-1))*1000
        row[name+"_projected_jerk_p95_px_frame3"] = p95(np.linalg.norm(np.diff(xy[:,ids],n=3,axis=0),axis=-1))
    row["root_translation_jerk_p95_mm_frame3"] = p95(np.linalg.norm(np.diff(effective_root,n=3,axis=0),axis=-1))*1000
    tips = [95,99,103,107,111,116,120,124,128,132]
    row["finger_jerk_p95_mm_frame3"] = p95(np.linalg.norm(np.diff(hands[:,tips]+c[:,None],n=3,axis=0),axis=-1))*1000
    row["finger_projected_jerk_p95_px_frame3"] = p95(np.linalg.norm(np.diff(projected[:,tips]*PIXELS,n=3,axis=0),axis=-1))
    for axis,name in enumerate("xyz"):
        row["camera_range_"+name] = float(np.ptp(c[:,axis]))
        row["root_translation_range_"+name+"_m"] = float(np.ptp(effective_root[:,axis]))
    return row
