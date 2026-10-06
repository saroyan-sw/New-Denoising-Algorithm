from __future__ import annotations

import numpy as np

from .operators import convolve_reflect, gaussian_blur


Array = np.ndarray


def gaussian_noise(image: Array, sigma: float, rng: np.random.Generator) -> Array:
    noise = rng.normal(0.0, sigma, size=np.asarray(image).shape).astype(np.float32)
    return np.clip(np.asarray(image, dtype=np.float32) + noise, 0.0, 1.0)


def poisson_noise(image: Array, peak: float, rng: np.random.Generator) -> Array:
    if peak <= 0:
        raise ValueError("Poisson peak must be positive")
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.clip(rng.poisson(array * peak).astype(np.float32) / peak, 0.0, 1.0)


def salt_and_pepper(image: Array, amount: float, rng: np.random.Generator) -> Array:
    if not 0.0 <= amount <= 1.0:
        raise ValueError("Salt-and-pepper amount must be in [0, 1]")
    array = np.asarray(image, dtype=np.float32).copy()
    mask_shape = array.shape[:2] + ((1,) if array.ndim == 3 else ())
    sample = rng.random(mask_shape)
    array = np.where(sample < amount / 2.0, 0.0, array)
    array = np.where(sample > 1.0 - amount / 2.0, 1.0, array)
    return array.astype(np.float32)


def gaussian_blur_degradation(image: Array, sigma: float = 2.0) -> Array:
    return np.clip(gaussian_blur(image, sigma=sigma), 0.0, 1.0)


def horizontal_motion_blur(image: Array, length: int = 9) -> Array:
    if length < 1 or length % 2 == 0:
        raise ValueError("Motion length must be a positive odd integer")
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return np.clip(convolve_reflect(image, kernel), 0.0, 1.0)


def standard_suite(image: Array, seed: int) -> dict[str, Array]:
    rng = np.random.default_rng(seed)
    return {
        "gaussian_0.05": gaussian_noise(image, 0.05, rng),
        "gaussian_0.10": gaussian_noise(image, 0.10, rng),
        "poisson_peak_50": poisson_noise(image, 50.0, rng),
        "salt_pepper_0.05": salt_and_pepper(image, 0.05, rng),
        "gaussian_blur_2.0": gaussian_blur_degradation(image, 2.0),
        "motion_blur_9": horizontal_motion_blur(image, 9),
    }

