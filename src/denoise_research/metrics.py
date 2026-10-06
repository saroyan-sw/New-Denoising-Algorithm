from __future__ import annotations

import math

import numpy as np

from .operators import gaussian_blur


Array = np.ndarray


def _pair(reference: Array, estimate: Array) -> tuple[Array, Array]:
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    if ref.shape != est.shape:
        raise ValueError(f"Metric inputs must have identical shapes: {ref.shape} != {est.shape}")
    return ref, est


def mse(reference: Array, estimate: Array) -> float:
    ref, est = _pair(reference, estimate)
    return float(np.mean((ref - est) ** 2))


def mae(reference: Array, estimate: Array) -> float:
    ref, est = _pair(reference, estimate)
    return float(np.mean(np.abs(ref - est)))


def psnr(reference: Array, estimate: Array, data_range: float = 1.0) -> float:
    error = mse(reference, estimate)
    if error == 0.0:
        return math.inf
    return float(10.0 * math.log10((data_range**2) / error))


def ssim(reference: Array, estimate: Array, data_range: float = 1.0) -> float:
    """Mean channel-wise SSIM with the standard 11x11 Gaussian window."""
    ref, est = _pair(reference, estimate)
    if ref.ndim == 2:
        ref = ref[:, :, None]
        est = est[:, :, None]
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_ref = gaussian_blur(ref, sigma=1.5, size=11)
    mu_est = gaussian_blur(est, sigma=1.5, size=11)
    mu_ref2 = mu_ref**2
    mu_est2 = mu_est**2
    mu_cross = mu_ref * mu_est
    var_ref = gaussian_blur(ref**2, sigma=1.5, size=11) - mu_ref2
    var_est = gaussian_blur(est**2, sigma=1.5, size=11) - mu_est2
    covariance = gaussian_blur(ref * est, sigma=1.5, size=11) - mu_cross
    numerator = (2.0 * mu_cross + c1) * (2.0 * covariance + c2)
    denominator = (mu_ref2 + mu_est2 + c1) * (var_ref + var_est + c2)
    score = numerator / np.maximum(denominator, 1e-12)
    return float(np.mean(score))


def full_reference_metrics(reference: Array, estimate: Array) -> dict[str, float]:
    return {
        "mse": mse(reference, estimate),
        "mae": mae(reference, estimate),
        "psnr": psnr(reference, estimate),
        "ssim": ssim(reference, estimate),
    }

