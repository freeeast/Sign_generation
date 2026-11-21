"""Low-dimensional SMPL-X updates composed as local rotations."""
import numpy as np
from scipy.spatial.transform import Rotation

from .probe_3d import ROTATION_GROUPS, regular_axes


def rotation(vectors):
    return Rotation.from_rotvec(vectors.reshape(-1, 3))


def delta_kinematics(start, candidate):
    position, velocity, acceleration = [], [], []
    for field, before in start.items():
        updated = candidate[field]
        frames, joints = updated.shape[:2]
        delta = (rotation(before).inv() * rotation(updated)).as_rotvec().reshape(frames, joints, 3)
        steps = (rotation(delta[:-1]).inv() * rotation(delta[1:])).as_rotvec().reshape(frames-1, joints, 3)
        position.extend(np.linalg.norm(delta, axis=-1).ravel())
        velocity.extend(np.linalg.norm(steps, axis=-1).ravel())
        acceleration.extend(np.linalg.norm(np.diff(steps, axis=0), axis=-1).ravel())
    return {name: float(np.rad2deg(np.max(values))) for name, values in (
        ("correction_joint_deg", position),
        ("correction_velocity_deg_per_frame", velocity),
        ("correction_accel_deg_per_frame2", acceleration),
    )}


class MotionSpace:
    def __init__(self, fit, asset, cfg):
        self.fit, self.cfg = fit, cfg
        self.n = len(fit["body_pose"])
        self.group = {family: ROTATION_GROUPS[family] for family in cfg["families"]}
        self.means = {}
        with np.load(asset, allow_pickle=False) as data:
            self.means["left_hand_pose"] = data["hands_meanl"].reshape(1, -1, 3)
            self.means["right_hand_pose"] = data["hands_meanr"].reshape(1, -1, 3)
        self.original = {field: fit[field].reshape(self.n, -1, 3) + self.means.get(field, 0)
                         for field in sorted({x for g in self.group.values() for x in g})}
        t = np.linspace(0, 1, self.n)
        taper = np.sin(np.pi*t)**2
        self.window = np.array([taper*(1-t), taper*t])
        self.window /= self.window.max(axis=1, keepdims=True)
        self.axis = {f: regular_axes(g, cfg["basis_seed"] + i*17)
                     for i, (f, g) in enumerate(self.group.items())}
        off_axes = {f: regular_axes(g, cfg["basis_seed"] + i*17 + 707)
                    for i, (f, g) in enumerate(self.group.items())}
        starter = self.compose(self.original, np.asarray(cfg["corruption_coeff_deg"]))
        for family, group in self.group.items():
            for field, ids in group.items():
                initial = starter[field][:, ids]
                delta = np.deg2rad(cfg["off_basis_deg"]) * taper[:, None, None] * off_axes[family][field][None]
                starter[field][:, ids] = (rotation(initial)*rotation(delta)).as_rotvec().reshape(initial.shape)
        self.initial = starter
        self.edited = {}
        for group in self.group.values():
            for field, joints in group.items():
                self.edited.setdefault(field, []).extend(joints)

    def compose(self, start, coefficients, *, window=None):
        window = self.window if window is None else window
        result = {field: val.copy() for field, val in start.items()}
        for i, (family, group) in enumerate(self.group.items()):
            dose = np.deg2rad(coefficients[2*i:2*i+2] @ window)
            for field, ids in group.items():
                current = start[field][:, ids]
                delta = dose[:, None, None] * self.axis[family][field][None]
                new = (rotation(current)*rotation(delta)).as_rotvec().reshape(current.shape)
                # Unwrap to the equivalent axis-angle branch nearest to the input.
                # Hand means are subtracted on export and restored by SMPL-X.
                norm = np.linalg.norm(new, axis=-1, keepdims=True)
                axis = np.divide(new, norm, out=np.zeros_like(new), where=norm>1e-12)
                alternate = new - 2*np.pi*axis
                closest = np.linalg.norm(alternate-current, axis=-1) < np.linalg.norm(new-current, axis=-1)
                result[field][:, ids] = np.where(closest[..., None], alternate, new)
        return result

    def export(self, tracks):
        updated = dict(self.fit)
        for field, indices in self.edited.items():
            value = self.fit[field].astype(np.float64).reshape(self.n, -1, 3)
            value[:, indices] = (tracks[field] - self.means.get(field, 0))[:, indices]
            updated[field] = value.reshape(self.fit[field].shape)
        return updated

    def candidate(self, coefficients):
        tracks = self.compose(self.initial, coefficients)
        kin = delta_kinematics(self.initial, tracks)
        kin["constraint_violations"] = sum(kin[k] > self.cfg["max_"+k] + 1e-6 for k in kin)
        return self.export(tracks), kin
