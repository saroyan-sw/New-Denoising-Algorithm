from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SpectralWeights:
    filter_weights: Array
    image_weights: Array
    iterations: int
    converged: bool


def _average_ranks(values: Array) -> Array:
    """Return zero-based average ranks with the smallest value ranked first."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def metric_to_affinity(
    scores: Array,
    *,
    lower_is_better: bool,
    mode: str = "rank",
    temperature: float = 1.0,
    floor: float = 1e-3,
) -> Array:
    """Convert an image-by-filter metric matrix into positive affinities.

    Conversion is performed independently within every image. This prevents easy
    images and metric scale from dominating before mutual-reinforcement ranking.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("scores must be a non-empty image-by-filter matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("scores must contain only finite values")
    if floor <= 0:
        raise ValueError("floor must be positive")

    quality = -matrix if lower_is_better else matrix
    if mode == "rank":
        result = np.empty_like(quality)
        denominator = max(1, matrix.shape[1] - 1)
        for index, row in enumerate(quality):
            badness_ranks = _average_ranks(-row)
            result[index] = 1.0 - badness_ranks / denominator
        return np.maximum(result, floor)
    if mode == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        centered = quality - np.median(quality, axis=1, keepdims=True)
        scale = np.median(np.abs(centered), axis=1, keepdims=True)
        scale = np.where(scale > 1e-12, scale, 1.0)
        logits = centered / (scale * temperature)
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
        result /= result.sum(axis=1, keepdims=True)
        return np.maximum(result, floor)
    raise ValueError("mode must be 'rank' or 'softmax'")


def combine_affinities(affinities: Sequence[Array]) -> Array:
    if not affinities:
        raise ValueError("At least one affinity matrix is required")
    matrices = [np.asarray(matrix, dtype=np.float64) for matrix in affinities]
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices):
        raise ValueError("All affinity matrices must have the same shape")
    combined = np.mean(np.stack(matrices, axis=0), axis=0)
    if np.any(combined <= 0) or not np.isfinite(combined).all():
        raise ValueError("Combined affinities must be finite and positive")
    return combined


def fit_spectral_weights(
    affinity: Array,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    damping: float = 0.95,
) -> SpectralWeights:
    """Fit HITS-like mutual-reinforcement weights on a bipartite score matrix.

    These are spectral consensus weights, not coefficients proven optimal for
    denoising error. Fit them on a calibration split and lock them for testing.
    """
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("affinity must be a non-empty image-by-filter matrix")
    if np.any(matrix <= 0) or not np.isfinite(matrix).all():
        raise ValueError("affinity must be finite and strictly positive")
    if max_iterations < 1 or tolerance <= 0 or not 0.0 < damping <= 1.0:
        raise ValueError("Invalid convergence parameters")

    image_count, filter_count = matrix.shape
    image_weights = np.full(image_count, 1.0 / image_count, dtype=np.float64)
    filter_weights = np.full(filter_count, 1.0 / filter_count, dtype=np.float64)
    converged = False
    for iteration in range(1, max_iterations + 1):
        proposed_filters = matrix.T @ image_weights
        proposed_filters /= proposed_filters.sum()
        proposed_filters = damping * proposed_filters + (1.0 - damping) / filter_count

        proposed_images = matrix @ proposed_filters
        proposed_images /= proposed_images.sum()
        proposed_images = damping * proposed_images + (1.0 - damping) / image_count

        change = max(
            float(np.max(np.abs(proposed_filters - filter_weights))),
            float(np.max(np.abs(proposed_images - image_weights))),
        )
        filter_weights, image_weights = proposed_filters, proposed_images
        if change < tolerance:
            converged = True
            break
    return SpectralWeights(filter_weights, image_weights, iteration, converged)


def fuse_candidates(candidates: Mapping[str, Array], weights: Mapping[str, float]) -> Array:
    if set(candidates) != set(weights):
        missing = sorted(set(candidates) - set(weights))
        extra = sorted(set(weights) - set(candidates))
        raise ValueError(f"Candidate/weight mismatch; missing={missing}, extra={extra}")
    names = list(candidates)
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("Fusion weights must be finite, nonnegative, and not all zero")
    values /= values.sum()
    stack = np.stack([np.asarray(candidates[name], dtype=np.float32) for name in names], axis=0)
    fused = np.tensordot(values.astype(np.float32), stack, axes=(0, 0))
    return np.clip(fused, 0.0, 1.0).astype(np.float32)
