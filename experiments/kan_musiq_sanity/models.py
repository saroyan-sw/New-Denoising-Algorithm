"""Prototype: a masked image encoder/decoder and conditional spline noise prior."""

import torch
from torch import nn
from efficient_kan import KAN


class MaskedImageKAN(nn.Module):
    def __init__(self, context_dim=4, condition_noise=False):
        super().__init__()
        self.condition_noise = condition_noise
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
        )
        self.context = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(16, context_dim), nn.Tanh())
        self.decoder = nn.Sequential(nn.Conv2d(16 + context_dim, 16, 3, padding=1),
                                     nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
        self.noise_prior = KAN([1 + (context_dim if condition_noise else 0), 8, 8, 1], grid_size=32, spline_order=3)

    def forward(self, noisy, mask):
        # Every image-dependent path excludes held-out values, including context.
        masked = noisy * (1 - mask)
        features = self.encoder(torch.cat([masked, mask], dim=1))
        context = self.context(features)
        spatial_context = context[:, :, None, None].expand(-1, -1, *noisy.shape[-2:])
        prediction = self.decoder(torch.cat([features, spatial_context], dim=1))
        return prediction, context

    def sample_noise(self, uniform, context):
        batch, samples, _ = uniform.shape
        code = context[:, None, :].expand(-1, samples, -1)
        inputs = torch.cat([2 * uniform - 1, code], dim=-1) if self.condition_noise else 2 * uniform - 1
        return self.noise_prior(inputs.reshape(batch * samples, -1)).reshape(batch, samples)


def checker_mask(noisy, partition=0):
    yy, xx = torch.meshgrid(torch.arange(noisy.shape[-2], device=noisy.device),
                            torch.arange(noisy.shape[-1], device=noisy.device), indexing="ij")
    membership = (yy % 2) * 2 + xx % 2
    return (membership == partition).to(noisy.dtype)[None, None].expand_as(noisy)


def blind_loss(prediction, noisy, mask):
    return ((prediction - noisy).square() * mask).sum() / mask.sum()
