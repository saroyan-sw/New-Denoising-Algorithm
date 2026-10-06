from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils import spectral_norm
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None
    nn = None
    F = None
    spectral_norm = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "The differentiable baseline requires PyTorch. "
            "Install the optional dependencies with: pip install -e .[torch]"
        )


if nn is not None:

    class ContrastivePatchGAN(nn.Module):
        """Architecture-compatible copy of the archived clean-domain discriminator."""

        def __init__(self, input_channels: int = 3):
            super().__init__()
            self.layer1 = nn.Sequential(
                spectral_norm(nn.Conv2d(input_channels, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True)
            )
            self.layer2 = nn.Sequential(
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2, True)
            )
            self.layer3 = nn.Sequential(
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2, True)
            )
            self.layer4 = nn.Sequential(
                spectral_norm(nn.Conv2d(256, 512, 4, 1, 1)), nn.LeakyReLU(0.2, True)
            )
            self.layer5 = spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))

        def forward(self, x: Any) -> tuple[Any, list[Any]]:
            feat1 = self.layer1(x)
            feat2 = self.layer2(feat1)
            feat3 = self.layer3(feat2)
            output = self.layer5(self.layer4(feat3))
            return output, [feat1, feat2, feat3]


    class MultiScaleDiscriminator(nn.Module):
        def __init__(self, input_channels: int = 3):
            super().__init__()
            self.discriminator_1x = ContrastivePatchGAN(input_channels)
            self.discriminator_half = ContrastivePatchGAN(input_channels)
            self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)

        def forward(self, x: Any) -> tuple[tuple[Any, Any], tuple[list[Any], list[Any]]]:
            out_1x, feats_1x = self.discriminator_1x(x)
            out_half, feats_half = self.discriminator_half(self.downsample(x))
            return (out_1x, out_half), (feats_1x, feats_half)


    def _gaussian_kernel(size: int, sigma: float, channels: int) -> Any:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        profile = torch.exp(-(coords**2) / (2.0 * sigma**2))
        profile /= profile.sum()
        kernel = profile[:, None] * profile[None, :]
        return kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)


    class DifferentiableBandFusion(nn.Module):
        """Three complementary bands with positive weights averaging to one.

        Internally the weights sum to three, so zero logits produce weights
        [1, 1, 1] and exactly reconstruct the input rather than one third of it.
        """

        def __init__(self, channels: int = 3, fine_sigma: float = 1.0, coarse_sigma: float = 2.0):
            super().__init__()
            if not 0 < fine_sigma < coarse_sigma:
                raise ValueError("Expected 0 < fine_sigma < coarse_sigma")
            fine_size = 2 * max(1, int(3 * fine_sigma + 0.999)) + 1
            coarse_size = 2 * max(1, int(3 * coarse_sigma + 0.999)) + 1
            self.register_buffer("fine_kernel", _gaussian_kernel(fine_size, fine_sigma, channels))
            self.register_buffer("coarse_kernel", _gaussian_kernel(coarse_size, coarse_sigma, channels))
            self.logits = nn.Parameter(torch.zeros(3))
            self.channels = channels

        @staticmethod
        def _blur(x: Any, kernel: Any) -> Any:
            pad = kernel.shape[-1] // 2
            padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
            return F.conv2d(padded, kernel, groups=x.shape[1])

        def bands(self, x: Any) -> tuple[Any, Any, Any]:
            fine = self._blur(x, self.fine_kernel)
            coarse = self._blur(x, self.coarse_kernel)
            return coarse, fine - coarse, x - fine

        def normalized_weights(self) -> Any:
            return 3.0 * torch.softmax(self.logits, dim=0)

        def forward(self, x: Any) -> Any:
            weights = self.normalized_weights()
            low, mid, high = self.bands(x)
            return weights[0] * low + weights[1] * mid + weights[2] * high


def load_clean_discriminator(checkpoint: str | Path, device: str | None = None) -> Any:
    require_torch()
    use_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiScaleDiscriminator().to(use_device)
    state = torch.load(Path(checkpoint), map_location=use_device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def optimize_with_frozen_prior(
    image: Any,
    discriminator: Any,
    steps: int = 150,
    learning_rate: float = 0.01,
    fidelity_weight: float = 10.0,
    mean_weight: float = 5.0,
    fine_sigma: float = 1.0,
    coarse_sigma: float = 2.0,
    extra_loss: Callable[[Any, Any], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Optimize only three fusion logits for one image.

    The L1 and mean terms reproduce the strongest later experimental setup. They
    are input-fidelity regularizers, not a physical noise likelihood.
    """
    require_torch()
    model = DifferentiableBandFusion(
        channels=image.shape[1],
        fine_sigma=fine_sigma,
        coarse_sigma=coarse_sigma,
    ).to(image.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(image)
        predictions, _ = discriminator(2.0 * reconstruction - 1.0)
        adversarial = sum(F.mse_loss(prediction, torch.ones_like(prediction)) for prediction in predictions)
        fidelity = F.l1_loss(reconstruction, image)
        mean = torch.abs(reconstruction.mean() - image.mean())
        loss = adversarial + fidelity_weight * fidelity + mean_weight * mean
        if extra_loss is not None:
            loss = loss + extra_loss(reconstruction, image)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    with torch.no_grad():
        result = model(image).clamp(0.0, 1.0)
        weights = model.normalized_weights().detach().cpu().tolist()
    return result, {"loss": history, "weights": weights}
