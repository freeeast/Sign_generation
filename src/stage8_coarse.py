"""Coarse monocular SMPL-X arm IK from real 2D keypoints with neutral fingers."""
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from smplx.lbs import batch_rodrigues
import torch

from .smplx_to_pose import load_model
from .stage7_feedback import valid_pose

# SMPL-X forearm chain and index/middle/ring MCPs, matched to COCO-WholeBody.
EDITED = np.arange(16, 22)
JOINTS = np.r_[16:22, 25, 28, 34, 40, 43, 49]
OBSERVATIONS = np.array([5, 6, 7, 8, 91, 112, 96, 100, 104, 117, 121, 125])


def coarse_fit(fit, asset, camera, xy, score, size, config):
    model = load_model(Path(asset).resolve(), "neutral", "model", "cpu",
                       (("num_betas", fit["betas"].shape[-1]),
                        ("num_expression_coeffs", fit["expression"].shape[-1])))
    n = len(xy)
    sampled = np.arange(0, n, config["fit_stride"])
    k = max(4, int(np.ceil((n - 1) / config["knot_spacing_frames"])) + 1)
    interpolation = torch.tensor(CubicSpline(np.linspace(0, n-1, k), np.eye(k),
                                              bc_type="natural")(np.arange(n)), dtype=torch.float32)
    to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
    # The shape and expression offset of rest joints can be computed without skinning vertices.
    with torch.no_grad():
        rest = model.J_regressor @ model.v_template
        shape_basis = torch.einsum("jv,vck->jck", model.J_regressor, model.shapedirs)
        expr_basis = torch.einsum("jv,vck->jck", model.J_regressor, model.expr_dirs)
        rest = (rest + torch.einsum("tk,jck->tjc", to_tensor(fit["betas"][sampled]), shape_basis)
                + torch.einsum("tk,jck->tjc", to_tensor(fit["expression"][sampled]), expr_basis))
    parents = model.parents.tolist()
    active = set(JOINTS.tolist())
    for j in JOINTS:
        while parents[j] >= 0:
            j = parents[j]
            active.add(j)
    template = np.zeros((len(sampled), 55, 3), np.float32)
    template[:, 0] = fit["global_orient"][sampled]
    template[:, 1:22] = fit["body_pose"][sampled].reshape(-1, 21, 3)
    # Arm rotations are reset and estimated anew. Collars use a static clip-level posture.
    template[:, [13, 14]] = np.median(template[:, [13, 14]], axis=0)
    template[:, EDITED] = 0
    base = to_tensor(template)
    radii = to_tensor(np.deg2rad(config["rotation_radius_deg"]))[None, :, None]
    frame_valid = valid_pose(xy, score, .3)[sampled][:, OBSERVATIONS]
    weight = to_tensor(np.clip(score[sampled][:, OBSERVATIONS], 0, 1) * frame_valid)
    weight[:, 6:] *= config["palm_weight"]
    target = to_tensor(xy[sampled][:, OBSERVATIONS])
    ctrans, affine = to_tensor(fit["cam_trans"][sampled, None]), to_tensor(camera)
    normalizer = to_tensor(np.asarray(size) / 100.)
    parameters = torch.nn.Parameter(torch.zeros(k, len(EDITED), 3))
    optimizer = torch.optim.Adam([parameters], lr=config["lr"])

    def controls(alpha):
        u = torch.einsum("tk,kjc->tjc", alpha, parameters)
        return u / torch.sqrt(1 + (u*u).sum(dim=-1, keepdim=True)) * radii

    def projection(ctrl):
        rotations = base.clone()
        rotations[:, EDITED] = ctrl
        mats = batch_rodrigues(rotations.reshape(-1, 3)).reshape(len(sampled), 55, 3, 3)
        global_r, positions = {}, {}
        for j in sorted(active):
            p = parents[j]
            if p < 0:
                global_r[j], positions[j] = mats[:, j], rest[:, j]
            else:
                positions[j] = positions[p] + (global_r[p] @ (rest[:, j] - rest[:, p])[..., None])[..., 0]
                global_r[j] = global_r[p] @ mats[:, j]
        xyz = torch.stack([positions[j] for j in JOINTS], dim=1) + ctrans
        projected = xyz[..., :2] / xyz[..., 2:].clamp_min(.1) * to_tensor([5000/192, 5000/256]) + .5
        return projected @ affine[:, :2].T + affine[:, 2]

    trace = []
    for step in range(config["steps"] + 1):
        optimizer.zero_grad()
        ctrl = controls(interpolation[sampled])
        residual = (projection(ctrl) - target) * normalizer
        robust = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual),
                                                  beta=.10, reduction="none").sum(-1)
        observation = (robust*weight).sum() / weight.sum().clamp_min(1)
        prior = config["rotation_weight"] * (ctrl*ctrl).mean()
        reg = config["knot_curvature_weight"] * parameters.diff(n=2, dim=0).square().mean()
        loss = observation + prior + reg
        if step % 10 == 0 or step == config["steps"]:
            trace.append(dict(step=step, objective=float(loss.detach()), observation=float(observation.detach())))
        if step < config["steps"]:
            loss.backward()
            optimizer.step()
    result = dict(fit)
    pose = fit["body_pose"].copy().reshape(n, 21, 3)
    pose[:, [12, 13]] = np.median(pose[:, [12, 13]], axis=0)
    pose[:, EDITED-1] = controls(interpolation).detach().numpy()
    result["body_pose"] = pose.reshape(n, 63)
    for key in ("left_hand_pose", "right_hand_pose"):
        result[key] = np.zeros((n, 45), dtype=np.float32)  # SMPL-X learned mean hand
    if not np.isfinite(result["body_pose"]).all():
        raise ValueError("Nonfinite coarse inverse kinematics")
    return result, trace, dict(knots=k, fitted_frames=len(sampled), observed_landmarks=int(frame_valid.sum()))
