"""Noise-to-motion rectified flow in the existing standardized rot6d coordinates."""
import math

import torch

from experiments.generative_motion.diffusion import GaussianDDPM
from experiments.generative_motion.motion import FEATURES


class RectifiedFlow:
    def __init__(self, steps, device, temporal_rho=0.0):
        if not 0 <= temporal_rho < 1:
            raise ValueError('temporal_rho must be in [0, 1)')
        self.temporal_rho = temporal_rho
        self.steps = steps
        self.device = device

    def source(self, shape, generator):
        noise = torch.randn(shape, device=self.device, generator=generator)
        if self.temporal_rho == 0:
            return noise
        frames = shape[1]
        frame = torch.arange(frames, device=self.device)
        lag = frame[:, None] - frame[None, :]
        rho = self.temporal_rho
        factor = torch.tril(rho ** lag.clamp_min(0)).to(noise.dtype)
        # Stationary Gaussian initialized at marginal variance one.
        factor[:, 1:] *= math.sqrt(1 - rho * rho)
        return factor @ noise

    def loss(self, model, clean, generator):
        noise = self.source(clean.shape, generator)
        time = torch.rand(len(clean), device=self.device, generator=generator)
        interpolated = noise.lerp(clean, time[:, None, None])
        velocity = model(interpolated, 100 * time)
        return (velocity - (clean - noise)).square().mean()

    @torch.inference_mode()
    def sample(self, model, frames, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        motion = self.source((1, frames, FEATURES), generator)
        for step in range(self.steps):
            time = motion.new_full((1,), 100 * step / self.steps)
            motion = motion + model(motion, time) / self.steps
        return motion[0]


def build_process(config, device):
    name = config.get("process", "ddpm")
    if name == "flow":
        return RectifiedFlow(config["diffusion_steps"], device, config.get("temporal_rho", 0.0))
    if name == "ddpm":
        return GaussianDDPM(config["diffusion_steps"], device, config.get("prediction", "noise"),
                            config.get("velocity_weight", 0.0))
    raise ValueError(f"Unknown generation process: {name}")
