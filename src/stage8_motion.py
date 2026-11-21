"""Side-specific, time-local finger controls for the second imitation phase."""
import numpy as np

from .stage5_motion import MotionSpace, delta_kinematics, rotation


class FingerWindowSpace(MotionSpace):
    """Keep the original six-axis approach step, add hand-specific refinement."""
    def __init__(self, fit, asset, cfg):
        super().__init__(fit, asset, cfg)
        t = np.linspace(0, 1, self.n)
        taper = np.sin(np.pi*t)**2
        self.hand_windows = np.stack([taper*np.exp(-.5*((t-c)/.18)**2) for c in (.2,.5,.8)])
        self.hand_windows /= self.hand_windows.max(axis=1,keepdims=True)

    def candidate(self, coefficients):
        tracks = self.compose(self.initial, coefficients[:6])
        for j,key in enumerate(("left_hand_pose","right_hand_pose")):
            mean = self.means[key][0]
            norm = np.linalg.norm(mean,axis=-1,keepdims=True)
            axes = mean/np.maximum(norm, 1e-7)
            # Suppress axes whose SMPL-X rest-to-mean change is nearly zero.
            axes *= np.minimum(norm/.25, 1)
            strength = np.deg2rad(coefficients[6+3*j:9+3*j] @ self.hand_windows)
            delta = strength[:,None,None]*axes[None]
            current = tracks[key]
            tracks[key] = (rotation(current)*rotation(delta)).as_rotvec().reshape(current.shape)
        limits = delta_kinematics(self.initial, tracks)
        limits["constraint_violations"] = sum(limits[k] > self.cfg["max_"+k] + 1e-6 for k in limits)
        return self.export(tracks), limits
