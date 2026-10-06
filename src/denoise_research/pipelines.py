from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

import numpy as np

from .config import ComplementaryBandConfig, PaperInspiredConfig
from .operators import bilateral_filter, edge_enhance, gabor_bank_energy, gaussian_blur


Array = np.ndarray
Objective = Callable[[Array], float]


@dataclass(frozen=True)
class PaperComponents:
    low: Array
    enhanced: Array
    texture: Array
    edge_mask: Array


class PaperInspiredPipeline:
    """Stable reconstruction of the non-learning Self-Fuse feature block.

    This is intentionally not a bit-for-bit copy of the archived code. It keeps
    signed Gabor responses in floating point, removes duplicate orientations,
    uses the bilateral image as the edge detector's preconditioner, and selects
    one best fusion rather than averaging every candidate that passes two gates.
    """

    def __init__(self, config: PaperInspiredConfig | None = None):
        self.config = config or PaperInspiredConfig()

    def components(self, image: Array) -> PaperComponents:
        x = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
        cfg = self.config
        low = bilateral_filter(
            x,
            radius=cfg.bilateral_radius,
            sigma_spatial=cfg.bilateral_sigma_spatial,
            sigma_range=cfg.bilateral_sigma_range,
        )
        enhanced, edge_mask = edge_enhance(
            x,
            edge_source=low,
            threshold_std=cfg.edge_threshold_std,
            strength=cfg.edge_strength,
        )
        texture = gabor_bank_energy(
            x,
            orientations=cfg.gabor_orientations,
            size=cfg.gabor_kernel_size,
            sigma=cfg.gabor_sigma,
            wavelength=cfg.gabor_wavelength,
            aspect_ratio=cfg.gabor_aspect_ratio,
        )
        return PaperComponents(low=low, enhanced=enhanced, texture=texture, edge_mask=edge_mask)

    @staticmethod
    def fuse(components: PaperComponents, weights: dict[str, float]) -> Array:
        names = ("low", "enhanced", "texture")
        values = np.array([float(weights[name]) for name in names], dtype=np.float32)
        if np.any(values < 0) or float(values.sum()) <= 0:
            raise ValueError("Paper-inspired fusion weights must be nonnegative and not all zero")
        values /= values.sum()
        result = (
            values[0] * components.low
            + values[1] * components.enhanced
            + values[2] * components.texture
        )
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def run(self, image: Array) -> Array:
        return self.fuse(self.components(image), self.config.fusion_weights)

    def grid_search(
        self,
        image: Array,
        objective: Objective,
        values: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> tuple[Array, dict[str, float], float]:
        components = self.components(image)
        best_image: Array | None = None
        best_weights: dict[str, float] | None = None
        best_score = float("inf")
        for low, enhanced, texture in product(values, repeat=3):
            if low + enhanced + texture <= 0:
                continue
            weights = {"low": low, "enhanced": enhanced, "texture": texture}
            candidate = self.fuse(components, weights)
            score = float(objective(candidate))
            if np.isfinite(score) and score < best_score:
                best_image, best_weights, best_score = candidate, weights, score
        if best_image is None or best_weights is None:
            raise RuntimeError("The fusion objective did not return a finite score")
        return best_image, best_weights, best_score


def complementary_bands(
    image: Array,
    fine_sigma: float = 1.0,
    coarse_sigma: float = 2.0,
) -> tuple[Array, Array, Array]:
    """Return low, mid, and high bands whose exact sum is the input image."""
    if not 0 < fine_sigma < coarse_sigma:
        raise ValueError("Expected 0 < fine_sigma < coarse_sigma")
    x = np.asarray(image, dtype=np.float32)
    coarse = gaussian_blur(x, sigma=coarse_sigma)
    fine = gaussian_blur(x, sigma=fine_sigma)
    low = coarse
    mid = fine - coarse
    high = x - fine
    return low.astype(np.float32), mid.astype(np.float32), high.astype(np.float32)


class ComplementaryBandFusion:
    """Corrected form of the later three-scalar differentiable baseline.

    Unit weights reconstruct the input. This corrects the archived implementation,
    where forcing the three coefficients to sum to one made equal weights output
    one third of the input even though the component bands already summed to it.
    """

    def __init__(self, config: ComplementaryBandConfig | None = None):
        self.config = config or ComplementaryBandConfig()

    def run(self, image: Array, weights: dict[str, float] | None = None) -> Array:
        cfg = self.config
        low, mid, high = complementary_bands(image, cfg.fine_sigma, cfg.coarse_sigma)
        use = cfg.weights if weights is None else weights
        result = (
            float(use["low"]) * low
            + float(use["mid"]) * mid
            + float(use["high"]) * high
        )
        return np.clip(result, 0.0, 1.0).astype(np.float32)


def overlap_add_apply(
    image: Array,
    processor: Callable[[Array], Array],
    patch_size: int = 64,
    stride: int | None = None,
) -> Array:
    """Apply a patch processor with reflected context and Hann overlap-add."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or RGB image")
    if patch_size < 4:
        raise ValueError("patch_size must be at least 4")
    stride = patch_size // 2 if stride is None else stride
    if not 1 <= stride <= patch_size:
        raise ValueError("stride must be in [1, patch_size]")

    pad = patch_size // 2
    pad_spec = ((pad, pad), (pad, pad)) + (((0, 0),) if x.ndim == 3 else ())
    padded = np.pad(x, pad_spec, mode="reflect")
    output = np.zeros_like(padded, dtype=np.float32)
    weight = np.zeros_like(padded, dtype=np.float32)

    hann = np.hanning(patch_size).astype(np.float32)
    window_2d = np.maximum(np.outer(hann, hann), 1e-3)
    window = window_2d[:, :, None] if x.ndim == 3 else window_2d

    def positions(length: int) -> list[int]:
        last = length - patch_size
        values = list(range(0, last + 1, stride))
        if values[-1] != last:
            values.append(last)
        return values

    for y in positions(padded.shape[0]):
        for x_pos in positions(padded.shape[1]):
            patch = padded[y : y + patch_size, x_pos : x_pos + patch_size]
            processed = np.asarray(processor(patch), dtype=np.float32)
            if processed.shape != patch.shape:
                raise ValueError("Patch processor must preserve patch shape")
            output[y : y + patch_size, x_pos : x_pos + patch_size] += processed * window
            weight[y : y + patch_size, x_pos : x_pos + patch_size] += window

    reconstructed = output / np.maximum(weight, 1e-8)
    return reconstructed[pad:-pad, pad:-pad].astype(np.float32)

