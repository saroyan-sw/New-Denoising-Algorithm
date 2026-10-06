from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .operators import bilateral_filter, gaussian_blur, median_filter


Array = np.ndarray


@dataclass(frozen=True)
class FilterSpec:
    """Named deterministic candidate used by the spectral ensemble."""

    name: str
    apply: Callable[[Array], Array]


def default_filter_bank() -> tuple[FilterSpec, ...]:
    """Return a small, deliberately redundant denoising candidate bank.

    Identity is retained so a learned ensemble can decide not to smooth. Multiple
    strengths are included because the experiment ranks filter configurations,
    not just algorithm families.
    """
    return (
        FilterSpec("identity", lambda x: np.asarray(x, dtype=np.float32).copy()),
        FilterSpec("gaussian_sigma_0.6", lambda x: gaussian_blur(x, sigma=0.6)),
        FilterSpec("gaussian_sigma_1.0", lambda x: gaussian_blur(x, sigma=1.0)),
        FilterSpec("gaussian_sigma_1.5", lambda x: gaussian_blur(x, sigma=1.5)),
        FilterSpec("median_3", lambda x: median_filter(x, size=3)),
        FilterSpec("median_5", lambda x: median_filter(x, size=5)),
        FilterSpec(
            "bilateral_mild",
            lambda x: bilateral_filter(x, radius=2, sigma_spatial=1.5, sigma_range=0.08),
        ),
        FilterSpec(
            "bilateral_medium",
            lambda x: bilateral_filter(x, radius=3, sigma_spatial=2.0, sigma_range=0.12),
        ),
        FilterSpec(
            "bilateral_strong",
            lambda x: bilateral_filter(x, radius=3, sigma_spatial=2.5, sigma_range=0.18),
        ),
    )


def apply_filter_bank(
    image: Array,
    bank: tuple[FilterSpec, ...] | None = None,
    clip_output: bool = True,
) -> dict[str, Array]:
    specs = default_filter_bank() if bank is None else bank
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Filter names must be unique")
    outputs: dict[str, Array] = {}
    for spec in specs:
        result = np.asarray(spec.apply(image), dtype=np.float32)
        if result.shape != np.asarray(image).shape:
            raise ValueError(f"Filter {spec.name!r} changed image shape")
        outputs[spec.name] = np.clip(result, 0.0, 1.0) if clip_output else result
    return outputs


def select_filter_bank(names: list[str] | tuple[str, ...]) -> tuple[FilterSpec, ...]:
    available = {spec.name: spec for spec in default_filter_bank()}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Unknown filters in saved weights: {missing}")
    return tuple(available[name] for name in names)
