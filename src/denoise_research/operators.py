from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


Array = np.ndarray


def _as_float_image(image: Array) -> Array:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected HxW or HxWxC image, got shape {array.shape}")
    return array


def convolve_reflect(image: Array, kernel: Array) -> Array:
    """Cross-correlate an image with an odd 2-D kernel using reflected borders."""
    array = _as_float_image(image)
    filt = np.asarray(kernel, dtype=np.float32)
    if filt.ndim != 2 or filt.shape[0] % 2 == 0 or filt.shape[1] % 2 == 0:
        raise ValueError("Kernel must be a two-dimensional array with odd dimensions")
    pad_y, pad_x = filt.shape[0] // 2, filt.shape[1] // 2
    pad_spec = ((pad_y, pad_y), (pad_x, pad_x))
    if array.ndim == 3:
        pad_spec += ((0, 0),)
    padded = np.pad(array, pad_spec, mode="reflect")
    windows = sliding_window_view(padded, filt.shape, axis=(0, 1))
    if array.ndim == 2:
        return np.einsum("hwkl,kl->hw", windows, filt, optimize=True).astype(np.float32)
    return np.einsum("hwckl,kl->hwc", windows, filt, optimize=True).astype(np.float32)


def gaussian_kernel(size: int, sigma: float) -> Array:
    if size < 1 or size % 2 == 0:
        raise ValueError("Gaussian kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    coords = np.arange(size, dtype=np.float32) - size // 2
    kernel_1d = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_blur(image: Array, sigma: float, size: int | None = None) -> Array:
    if size is None:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        size = 2 * radius + 1
    return convolve_reflect(image, gaussian_kernel(size, sigma))


def box_blur(image: Array, size: int = 3) -> Array:
    if size < 1 or size % 2 == 0:
        raise ValueError("Box size must be a positive odd integer")
    kernel = np.full((size, size), 1.0 / (size * size), dtype=np.float32)
    return convolve_reflect(image, kernel)


def median_filter(image: Array, size: int = 3) -> Array:
    """Apply a channel-wise median filter with reflected borders."""
    array = _as_float_image(image)
    if size < 1 or size % 2 == 0:
        raise ValueError("Median size must be a positive odd integer")
    radius = size // 2
    pad_spec = ((radius, radius), (radius, radius))
    if array.ndim == 3:
        pad_spec += ((0, 0),)
    padded = np.pad(array, pad_spec, mode="reflect")
    patches = sliding_window_view(padded, (size, size), axis=(0, 1))
    return np.median(patches, axis=(-2, -1)).astype(np.float32)


def bilateral_filter(
    image: Array,
    radius: int = 4,
    sigma_spatial: float = 3.0,
    sigma_range: float = 0.10,
) -> Array:
    """Reference bilateral filter for float images in [0, 1]."""
    array = _as_float_image(image)
    if radius < 1 or sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("Bilateral radius and sigmas must be positive")
    size = 2 * radius + 1
    pad_spec = ((radius, radius), (radius, radius))
    if array.ndim == 3:
        pad_spec += ((0, 0),)
    padded = np.pad(array, pad_spec, mode="reflect")
    patches = sliding_window_view(padded, (size, size), axis=(0, 1))

    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    spatial = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_spatial**2))

    if array.ndim == 2:
        delta2 = (patches - array[:, :, None, None]) ** 2
        weights = spatial[None, None, :, :] * np.exp(-delta2 / (2.0 * sigma_range**2))
        normalizer = weights.sum(axis=(-2, -1)) + 1e-12
        result = (patches * weights).sum(axis=(-2, -1)) / normalizer
    else:
        delta2 = ((patches - array[:, :, :, None, None]) ** 2).mean(axis=2)
        weights = spatial[None, None, :, :] * np.exp(-delta2 / (2.0 * sigma_range**2))
        normalizer = weights.sum(axis=(-2, -1)) + 1e-12
        result = np.einsum("hwckl,hwkl->hwc", patches, weights, optimize=True)
        result /= normalizer[:, :, None]
    return result.astype(np.float32)


def sobel_gradients(image: Array) -> tuple[Array, Array]:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    kernel_y = kernel_x.T
    return convolve_reflect(image, kernel_x), convolve_reflect(image, kernel_y)


def sobel_magnitude(image: Array) -> Array:
    grad_x, grad_y = sobel_gradients(image)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if magnitude.ndim == 3:
        magnitude = np.sqrt(np.mean(magnitude**2, axis=2, keepdims=True))
    return magnitude.astype(np.float32)


def gabor_kernel(
    size: int,
    sigma: float,
    theta: float,
    wavelength: float,
    aspect_ratio: float = 0.5,
    phase: float = 0.0,
) -> Array:
    if size < 3 or size % 2 == 0:
        raise ValueError("Gabor kernel size must be an odd integer of at least 3")
    if min(sigma, wavelength, aspect_ratio) <= 0:
        raise ValueError("Gabor sigma, wavelength, and aspect ratio must be positive")
    coords = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    x_theta = xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
    envelope = np.exp(-(x_theta**2 + (aspect_ratio**2) * y_theta**2) / (2.0 * sigma**2))
    carrier = np.cos((2.0 * math.pi * x_theta / wavelength) + phase)
    kernel = envelope * carrier
    kernel -= kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-12
    return kernel.astype(np.float32)


def gabor_bank_energy(
    image: Array,
    orientations: int = 8,
    size: int = 9,
    sigma: float = 2.0,
    wavelength: float = 4.0,
    aspect_ratio: float = 0.5,
) -> Array:
    """Aggregate zero-mean oriented responses as RMS energy without uint8 clipping."""
    if orientations < 1:
        raise ValueError("At least one Gabor orientation is required")
    responses = []
    for index in range(orientations):
        theta = math.pi * index / orientations
        kernel = gabor_kernel(size, sigma, theta, wavelength, aspect_ratio)
        responses.append(convolve_reflect(image, kernel))
    energy = np.sqrt(np.mean(np.square(np.stack(responses, axis=0)), axis=0))
    scale = float(np.percentile(energy, 99.0))
    if scale <= 1e-12:
        return np.zeros_like(energy, dtype=np.float32)
    return np.clip(energy / scale, 0.0, 1.0).astype(np.float32)


def edge_enhance(
    image: Array,
    edge_source: Array | None = None,
    threshold_std: float = 3.0,
    strength: float = 0.20,
) -> tuple[Array, Array]:
    """Enhance statistically strong Sobel responses and return image plus edge mask."""
    array = _as_float_image(image)
    source = array if edge_source is None else _as_float_image(edge_source)
    magnitude = sobel_magnitude(source)
    threshold = float(magnitude.mean() + threshold_std * magnitude.std())
    mask = (magnitude > threshold).astype(np.float32)
    enhanced = np.clip(array + strength * magnitude * mask, 0.0, 1.0)
    return enhanced.astype(np.float32), mask
