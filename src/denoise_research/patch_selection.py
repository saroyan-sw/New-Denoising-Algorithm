from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .filter_bank import FilterSpec
from .operators import bilateral_filter, gaussian_blur, median_filter
from .nr_metrics import MetricSpec
from . import torch_optional


Array = np.ndarray
PatchScorer = Callable[[Array], Mapping[str, float]]


@dataclass(frozen=True)
class PatchSelectionResult:
    image: Array
    selection_map: Array
    filter_names: tuple[str, ...]
    selected_patch_counts: dict[str, int]


class CleanDiscriminatorScorer:
    """Score patches with the project's archived pretrained clean discriminator.

    The discriminator's real/clean target is encoded as a higher PatchGAN logit.
    This is a learned naturalness prior, not a calibrated probability or a
    source-fidelity measure.
    """

    metric_specs = (MetricSpec("clean_discriminator", lower_is_better=False),)

    def __init__(self, discriminator: object, device: str | None = None):
        torch_optional.require_torch()
        self.torch = torch_optional.torch
        self.discriminator = discriminator
        self.device = self.torch.device(device or next(discriminator.parameters()).device)
        self.discriminator.eval()

    def __call__(self, image: Array) -> dict[str, float]:
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Expected HxW or HxWx3 image")
        tensor = self.torch.from_numpy(np.ascontiguousarray(array))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device).clamp(0.0, 1.0)
        with self.torch.no_grad():
            predictions, _ = self.discriminator(2.0 * tensor - 1.0)
            score = self.torch.stack([prediction.mean() for prediction in predictions]).mean()
        return {"clean_discriminator": float(score.cpu())}


def noise_specific_filter_bank(noise_type: str) -> tuple[FilterSpec, ...]:
    """Return classical candidates intended for the explicitly selected noise model."""
    identity = FilterSpec("identity", lambda x: np.asarray(x, dtype=np.float32).copy())
    if noise_type == "awgn":
        denoisers = (
            FilterSpec("gaussian_sigma_0.6", lambda x: gaussian_blur(x, sigma=0.6)),
            FilterSpec("gaussian_sigma_1.0", lambda x: gaussian_blur(x, sigma=1.0)),
            FilterSpec("gaussian_sigma_1.5", lambda x: gaussian_blur(x, sigma=1.5)),
            FilterSpec(
                "bilateral_mild",
                lambda x: bilateral_filter(x, radius=2, sigma_spatial=1.5, sigma_range=0.08),
            ),
            FilterSpec(
                "bilateral_medium",
                lambda x: bilateral_filter(x, radius=3, sigma_spatial=2.0, sigma_range=0.12),
            ),
        )
    elif noise_type in {"salt_pepper", "salt-and-pepper", "impulse"}:
        denoisers = (
            FilterSpec("median_3", lambda x: median_filter(x, size=3)),
            FilterSpec("median_5", lambda x: median_filter(x, size=5)),
        )
    else:
        raise ValueError("noise_type must be 'awgn' or 'salt_pepper'")
    return (identity, *denoisers)


def _positions(length: int, patch_size: int, stride: int) -> list[int]:
    last = length - patch_size
    values = list(range(0, last + 1, stride))
    if not values or values[-1] != last:
        values.append(last)
    return values


def _rank_affinities(
    scores: np.ndarray, specs: Sequence[MetricSpec]
) -> np.ndarray:
    """Convert incomparable metric values into per-patch candidate rank scores."""
    affinity = np.zeros(scores.shape[1], dtype=np.float64)
    for metric_index, spec in enumerate(specs):
        values = scores[metric_index]
        oriented = values if spec.lower_is_better else -values
        order = np.argsort(oriented, kind="stable")
        ranks = np.empty(len(order), dtype=np.float64)
        ranks[order] = np.arange(len(order), dtype=np.float64)
        if len(order) > 1:
            ranks /= len(order) - 1
        affinity += 1.0 - ranks
    return affinity / len(specs)


def select_patches(
    candidates: Mapping[str, Array],
    scorer: PatchScorer,
    metric_specs: Sequence[MetricSpec],
    *,
    patch_size: int = 96,
    stride: int | None = None,
) -> PatchSelectionResult:
    """Choose the best scored denoiser patch at each location and blend overlaps.

    Metrics are evaluated only on candidate outputs. No clean reference is used.
    Overlapping winning patches are Hann blended to reduce visible tile boundaries.
    """
    if not candidates:
        raise ValueError("At least one candidate image is required")
    if not metric_specs:
        raise ValueError("At least one no-reference metric is required")
    metric_names = [spec.name for spec in metric_specs]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("Metric names must be unique")
    names = tuple(candidates)
    if len(names) != len(set(names)):
        raise ValueError("Candidate names must be unique")
    images = [np.asarray(candidates[name], dtype=np.float32) for name in names]
    shape = images[0].shape
    if any(image.shape != shape for image in images):
        raise ValueError("All candidates must have identical shapes")
    if len(shape) not in (2, 3):
        raise ValueError("Expected grayscale or RGB images")
    if patch_size < 16:
        raise ValueError("patch_size must be at least 16 for patch quality assessment")
    stride = max(1, patch_size // 2) if stride is None else stride
    if not 1 <= stride <= patch_size:
        raise ValueError("stride must be in [1, patch_size]")

    height, width = shape[:2]
    pad = patch_size // 2
    pad_spec = ((pad, pad), (pad, pad)) + (((0, 0),) if len(shape) == 3 else ())
    mode = "reflect" if min(height, width) > 1 else "edge"
    padded = [np.pad(image, pad_spec, mode=mode) for image in images]
    padded_height, padded_width = padded[0].shape[:2]

    hann = np.hanning(patch_size).astype(np.float32)
    window_2d = np.maximum(np.outer(hann, hann), 1e-3)
    window = window_2d[:, :, None] if len(shape) == 3 else window_2d
    output = np.zeros_like(padded[0], dtype=np.float32)
    weights = np.zeros_like(padded[0], dtype=np.float32)
    selection_map = np.full((height, width), -1, dtype=np.int32)
    map_strength = np.full((height, width), -1.0, dtype=np.float32)
    selected_counts = {name: 0 for name in names}

    for top in _positions(padded_height, patch_size, stride):
        for left in _positions(padded_width, patch_size, stride):
            patches = [
                image[top : top + patch_size, left : left + patch_size]
                for image in padded
            ]
            scores = np.empty((len(metric_specs), len(names)), dtype=np.float64)
            for candidate_index, patch in enumerate(patches):
                values = scorer(patch)
                for metric_index, metric_name in enumerate(metric_names):
                    if metric_name not in values:
                        raise ValueError(f"Scorer did not return metric {metric_name!r}")
                    score = float(values[metric_name])
                    if not np.isfinite(score):
                        raise ValueError(f"Metric {metric_name!r} returned a non-finite score")
                    scores[metric_index, candidate_index] = score
            winner = int(np.argmax(_rank_affinities(scores, metric_specs)))
            selected_counts[names[winner]] += 1
            output[top : top + patch_size, left : left + patch_size] += patches[winner] * window
            weights[top : top + patch_size, left : left + patch_size] += window

            # Keep the strongest-window winner at each pixel for a dense selection map.
            image_top = max(0, top - pad)
            image_left = max(0, left - pad)
            image_bottom = min(height, top + patch_size - pad)
            image_right = min(width, left + patch_size - pad)
            patch_top = image_top - (top - pad)
            patch_left = image_left - (left - pad)
            patch_bottom = patch_top + image_bottom - image_top
            patch_right = patch_left + image_right - image_left
            local_strength = window_2d[patch_top:patch_bottom, patch_left:patch_right]
            region = map_strength[image_top:image_bottom, image_left:image_right]
            replace = local_strength > region
            region[replace] = local_strength[replace]
            labels = selection_map[image_top:image_bottom, image_left:image_right]
            labels[replace] = winner

    reconstructed = output / np.maximum(weights, 1e-8)
    result = reconstructed[pad : pad + height, pad : pad + width]
    return PatchSelectionResult(
        image=np.clip(result, 0.0, 1.0).astype(np.float32),
        selection_map=selection_map,
        filter_names=names,
        selected_patch_counts=selected_counts,
    )
