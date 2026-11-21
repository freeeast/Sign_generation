"""Unconditional Gaussian DDPM with a temporal convolutional denoiser."""
import math

import torch
from torch import nn

from experiments.generative_motion.motion import FEATURES


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv1d(width, width, 1),
        )
        self.time = nn.Linear(width, width)

    def forward(self, features, time):
        return features + self.layers(features + self.time(time)[..., None])


class Denoiser(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.input = nn.Conv1d(FEATURES, width, 1)
        self.blocks = nn.ModuleList(
            TemporalBlock(width, 2 ** (index % 6)) for index in range(depth)
        )
        self.output = nn.Conv1d(width, FEATURES, 1)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.register_buffer("frequencies", torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2)))

    def forward(self, motion, step):
        angles = step[:, None] * self.frequencies
        time = self.time(torch.cat((angles.sin(), angles.cos()), dim=-1))
        features = self.input(motion.transpose(1, 2))
        for block in self.blocks:
            features = block(features, time)
        return self.output(features).transpose(1, 2)


class GaussianDDPM:
    def __init__(self, steps, device, prediction='noise', velocity_weight=0.0):
        self.prediction = prediction
        self.velocity_weight = velocity_weight
        # Cosine cumulative schedule from Nichol & Dhariwal (2021).
        phase = (torch.arange(steps + 1, dtype=torch.float64) / steps + 0.008) / 1.008
        cosine = (phase * math.pi / 2).cos().square()
        self.beta = (1 - cosine[1:] / cosine[:-1]).clamp(max=0.999).float().to(device)
        self.alpha = 1 - self.beta
        self.cumulative = self.alpha.cumprod(0)

    def loss(self, model, clean, generator):
        step = torch.randint(len(self.beta), (len(clean),), device=clean.device, generator=generator)
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
        alpha = self.cumulative[step, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        prediction = model(noisy, step)
        if self.prediction == 'noise':
            return (prediction - noise).square().mean()
        pose_error = prediction - clean
        return pose_error.square().mean() + self.velocity_weight * pose_error.diff(dim=1).square().mean()

    @torch.inference_mode()
    def sample(self, model, frames, seed):
        device = self.beta.device
        rng = torch.Generator(device=device).manual_seed(seed)
        motion = torch.randn((1, frames, FEATURES), device=device, generator=rng)
        for i in reversed(range(len(self.beta))):
            step = torch.full((1,), i, device=device, dtype=torch.long)
            cumulative = self.cumulative[i]
            prediction = model(motion, step)
            if self.prediction == 'noise':
                prediction = (motion - (1 - cumulative).sqrt() * prediction) / cumulative.sqrt()
            # Clipping x0 in standardized coordinates limits extreme OOD values.
            clean = prediction.clamp(-5, 5)
            if i == 0:
                motion = clean
                continue
            previous = self.cumulative[i - 1]
            mean = (
                self.beta[i] * previous.sqrt() * clean
                + (1 - previous) * self.alpha[i].sqrt() * motion
            ) / (1 - cumulative)
            variance = self.beta[i] * (1 - previous) / (1 - cumulative)
            motion = mean + variance.sqrt() * torch.randn(motion.shape, device=device, generator=rng)
        return motion[0]
