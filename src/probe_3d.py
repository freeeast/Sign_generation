"""Run finite candidate comparisons on fitted 3D sign motion with a fixed camera."""
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .evaluate_feedback import write_csv
from .pose import FINGERS, LEFT, RIGHT
from .smplx_to_pose import forward
from .unisign import cosine
from .virtual_pose import TIPS, apply_affine, geometry_metrics, load_fit, pinhole

# Body indices omit the global root. Hand indices use the SMPL-X/MANO ordering.
ROTATION_GROUPS = {
    "finger_articulation": {"left_hand_pose": tuple(range(15)), "right_hand_pose": tuple(range(15))},
    "wrist_orientation": {"body_pose": (19, 20)},
    "wrist_position_via_elbows": {"body_pose": (17, 18)},
    "arm_trajectory": {"body_pose": (15, 16, 17, 18)},
    "local_head": {"body_pose": (11, 14)},
}
TEMPORAL_FIELDS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "jaw_pose")
BONE_PAIRS = [(5, 7), (7, 9), (6, 8), (8, 10), *[
    edge for start in (91, 112) for finger in range(5)
    for chain in [[start, *range(start + 1 + 4 * finger, start + 5 + 4 * finger)]]
    for edge in zip(chain, chain[1:])]]


def regular_axes(group, seed):
    gen = np.random.default_rng(seed)
    vectors = {}
    for field, ids in group.items():
        vector = gen.normal(size=(len(ids), 3))
        vectors[field] = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    return vectors


def rotate(track, ids, vectors, dose, window):
    frames = len(track)
    target = track.copy().reshape(frames, -1, 3)
    selected = target[:, ids]
    rotations = Rotation.from_rotvec(selected.reshape(-1, 3))
    offsets = np.deg2rad(dose) * window[:, None, None] * vectors[None, :, :]
    target[:, ids] = (rotations * Rotation.from_rotvec(offsets.reshape(-1, 3))).as_rotvec().reshape(selected.shape)
    return target.reshape(track.shape)


def warp(fit, shift):
    frames = len(fit["global_orient"])
    time = np.arange(frames)
    target = np.clip(time + shift * np.sin(2 * np.pi * time / max(1, frames - 1)), 0, frames - 1)
    changed = dict(fit)
    for field in TEMPORAL_FIELDS:
        old = fit[field].reshape(frames, -1, 3)
        curves = [Slerp(time, Rotation.from_rotvec(joint))(target).as_rotvec() for joint in old.transpose(1, 0, 2)]
        changed[field] = np.stack(curves, axis=1).reshape(fit[field].shape)
    changed["expression"] = np.stack([np.interp(target, time, x) for x in fit["expression"].T], axis=1)
    return changed


def candidates(fit, seed):
    frames = len(fit["global_orient"])
    window = np.sin(np.linspace(0, np.pi, frames)) ** 2
    for family, group in ROTATION_GROUPS.items():
        axes = regular_axes(group, seed)
        alt = regular_axes(group, seed + 1)
        for name, dose, directions in (
            ("dose3", 3, axes), ("dose8", 8, axes), ("corrupted16", 16, axes),
            ("restored_from16_to8", 8, axes), ("alternative_from16", 16, axes),
            ("negative3", -3, axes),
        ):
            motion = dict(fit)
            for field, ids in group.items():
                track = rotate(fit[field], ids, directions[field], dose, window)
                if name == "alternative_from16":
                    track = rotate(track, ids, alt[field], 8, window)
                motion[field] = track
            yield family, name, dose, "deg", motion
    # Restoration names reuse the same point on the dose curve. The alternative
    # keeps the peak displacement and changes the timing of this local comparison.
    for name, shift in (("dose1", 1), ("dose3", 3), ("corrupted7", 7),
                        ("restored_from7_to3", 3), ("alternative_reverse_shift", -7), ("negative1", -1)):
        yield "temporal_warp", name, shift, "frames", warp(fit, shift)


def diagnostics(points, baseline):
    first, second = np.array(BONE_PAIRS).T
    base_len = np.linalg.norm(baseline[:, first] - baseline[:, second], axis=-1)
    new_len = np.linalg.norm(points[:, first] - points[:, second], axis=-1)
    return dict(bone_length_drift_p95_m=float(np.percentile(np.abs(new_len - base_len), 95)),
                fingertip_and_joint_shift_p95_m=float(np.percentile(
                    np.linalg.norm((points - baseline)[:, FINGERS], axis=-1), 95)))


def finger_observations(synthetic, real, scores, wh):
    """Compare global fingertips and wrist-relative hand shape on disjoint frames."""
    wrist = np.r_[np.repeat(LEFT[0], 5), np.repeat(RIGHT[0], 5)]
    delta = (synthetic - real) * np.asarray(wh)
    handshape = np.linalg.norm(delta[:, TIPS] - delta[:, wrist], axis=-1)
    global_tip = np.linalg.norm(delta[:, TIPS], axis=-1)
    valid = (scores[:, TIPS] > .3) & (scores[:, wrist] > .3)

    def measure(values, group):
        data = values[group][valid[group]]
        return float(np.median(data)) if len(data) else None

    return dict(handshape_holdout_px=measure(handshape, slice(1, None, 2)),
                tip_geometry_A_px=measure(global_tip, slice(1, None, 4)),
                tip_geometry_B_px=measure(global_tip, slice(3, None, 4)),
                handshape_geometry_A_px=measure(handshape, slice(1, None, 4)),
                handshape_geometry_B_px=measure(handshape, slice(3, None, 4)))


def probe(model, model_dir, hand_mean, device, row, path, baseline, camera,
          real, real_score, wh, real_embedding, output):
    from .project_keypoints import plot_frames

    clip = row["clip_id"]
    fit = load_fit(path)
    if len(baseline) < 12:
        raise ValueError("A local probe requires at least 12 aligned frames")
    if row.get("rgb_fps") != 24:
        raise ValueError("Update time warp magnitudes for a non-24fps sequence")
    stored_camera = fit["cam_trans"]
    full = np.ones(real.shape[:2], np.float32)
    out = []
    poses = {}

    def score(family, case, dose, unit, points, mapped):
        xy = apply_affine(pinhole(points, stored_camera), camera)
        observation = model.infer(xy, full * mapped, [row["reference"]], max_frames=256)
        result = dict(clip=clip, family=family, case=case, magnitude=dose, magnitude_units=unit,
                      nll=observation["target_nll"][0],
                      cosine_to_real=cosine(observation["embedding"], real_embedding),
                      **geometry_metrics(points, xy, real, real_score, wh),
                      **finger_observations(xy, real, real_score, wh), **diagnostics(points, baseline))
        poses[family, case] = xy
        print("probe", clip, family, case, round(result["nll"], 4), flush=True)
        return result

    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch) / "motion.npz"
        base = score("baseline", "original", 0, "none", baseline, np.ones(133, bool))
        out.append(dict(base, delta_nll=0., geometry_delta_fingertips_px=0.))
        for group, name, dose, unit, pose in candidates(fit, seed=20261116):
            if name.startswith("restored"):
                reused = next(r for r in out if r["family"] == group and r["case"] == f"dose{dose}")
                out.append(dict(reused, case=name))
                poses[group, name] = poses[group, f"dose{dose}"]
                continue
            np.savez(temp, **pose)
            points, mapped = forward(temp, model_dir, "neutral", hand_mean, 64, device, False)
            result = score(group, name, dose, unit, points, mapped)
            holdout = result["fingertips_holdout_px"]
            delta = None if holdout is None or base["fingertips_holdout_px"] is None else holdout - base["fingertips_holdout_px"]
            out.append(dict(result, delta_nll=result["nll"] - base["nll"], geometry_delta_fingertips_px=delta))
    write_csv(output / f"{clip}_local_probes.csv", out)
    by_case = {(r["family"], r["case"]): r for r in out}
    ranks = []
    for family in (*ROTATION_GROUPS, "temporal_warp"):
        temporal = family == "temporal_warp"
        comparison = ("restored_from7_to3", "corrupted7", "alternative_reverse_shift", "dose1", "negative1") if temporal else (
            "restored_from16_to8", "corrupted16", "alternative_from16", "dose3", "negative3")
        restored, corrupt, alternative, positive, negative = [by_case[family, case] for case in comparison]
        slope = (positive["nll"] - negative["nll"]) / (2 * positive["magnitude"])
        for other in (corrupt, alternative):
            g1, g2 = restored["fingertips_holdout_px"], other["fingertips_holdout_px"]
            nll_delta = restored["nll"] - other["nll"]
            geometry_delta = None if g1 is None or g2 is None else g1 - g2
            # This is detector agreement; holdout ASL judgements are required to assess semantic quality.
            ranks.append(dict(clip=clip, family=family, comparison=other["case"], nll_restored_minus_other=nll_delta,
                              reprojection_restored_minus_other=geometry_delta,
                              matching_sign=None if geometry_delta is None or abs(geometry_delta) < .1 or abs(nll_delta) < 1e-4
                              else bool(np.sign(nll_delta) == np.sign(geometry_delta)),
                              nll_central_difference_per_unit=slope, finite_difference_units=positive["magnitude_units"]))
            if geometry_delta is not None and abs(geometry_delta) >= .1 and nll_delta * geometry_delta < 0:
                for candidate in (restored, other):
                    case = candidate["case"]
                    plot_frames(poses[family, case], full, output / f"{clip}_{family}_{case}.png")
    write_csv(output / f"{clip}_local_ranking.csv", ranks)
    plot_curves(out, output / f"{clip}_curves.png")


def plot_curves(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original = rows[0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for group, ax in zip((*ROTATION_GROUPS, "temporal_warp"), axes.flat):
        curves = [original, *[r for r in rows if r["family"] == group and
                             (r["case"].startswith("dose") or r["case"].startswith("corrupted"))]]
        ax.plot([r["magnitude"] for r in curves], [r["delta_nll"] for r in curves], "o-", color="#145970")
        ax.set(title=group.replace("_", " "), xlabel=curves[-1]["magnitude_units"], ylabel="ΔNLL")
        ax2 = ax.twinx()
        ax2.plot([r["magnitude"] for r in curves],
                 [np.nan if r["geometry_delta_fingertips_px"] is None else r["geometry_delta_fingertips_px"] for r in curves],
                 "s--", color="#c16338")
        ax2.set_ylabel("Δ fingertip error (px)", color="#a64e28")
    fig.savefig(path, dpi=150)
    plt.close(fig)
