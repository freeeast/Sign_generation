"""A nested 9-D controller with one additional smooth temporal mode per family."""
import numpy as np

from .stage5_motion import MotionSpace


class NestedSpace(MotionSpace):
    def __init__(self, fit, asset, cfg):
        super().__init__(fit,asset,cfg)
        t=np.linspace(0,1,self.n)
        bump=np.sin(np.pi*t)**2 * np.exp(-((t-.58)/.17)**2)
        bump-=self.window.T@np.linalg.lstsq(self.window.T,bump,rcond=None)[0]
        bump/=max(np.abs(bump).max(),1e-8)
        self.local=np.stack([bump,np.zeros_like(bump)])

    def compose(self, start, coefficients):
        coarse=super().compose(start,coefficients[:6])
        if len(coefficients)==6:
            return coarse
        fine=np.zeros(6)
        fine[::2]=coefficients[6:]
        # Same axis family and SO(3) composition, with one extra temporal shape.
        return super().compose(coarse,fine,window=self.local)
