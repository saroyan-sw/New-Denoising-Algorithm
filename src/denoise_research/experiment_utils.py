from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray


def random_crop(image: Array, size: int, rng: np.random.Generator) -> Array:
    """Select a reproducible square crop, or return the full image for size <= 0."""
    array = np.asarray(image)
    if size <= 0:
        return array.copy()
    height, width = array.shape[:2]
    if height < size or width < size:
        raise ValueError(f"Image {height}x{width} is smaller than crop size {size}")
    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    return array[top : top + size, left : left + size].copy()


def add_awgn(image: Array, sigma_255: float, rng: np.random.Generator) -> Array:
    """Add zero-mean AWGN without clipping the noisy observation."""
    if sigma_255 < 0:
        raise ValueError("Noise sigma must be nonnegative")
    noise = rng.normal(0.0, sigma_255 / 255.0, size=np.asarray(image).shape)
    return (np.asarray(image, dtype=np.float32) + noise).astype(np.float32)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
