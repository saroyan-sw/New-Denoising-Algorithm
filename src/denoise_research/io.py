from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an RGB image as float32 in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save a grayscale or RGB float image after safe clipping."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    encoded = np.round(array * 255.0).astype(np.uint8)
    mode = "L" if encoded.ndim == 2 else "RGB"
    Image.fromarray(encoded, mode=mode).save(target)


def image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)

