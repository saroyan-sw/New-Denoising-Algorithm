from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def comparison_grid(images: Mapping[str, np.ndarray], output: str | Path) -> None:
    """Save a labeled single-row comparison grid."""
    if not images:
        raise ValueError("At least one image is required")
    encoded: list[tuple[str, Image.Image]] = []
    for label, array in images.items():
        data = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
        data = np.round(data * 255.0).astype(np.uint8)
        mode = "L" if data.ndim == 2 else "RGB"
        encoded.append((label, Image.fromarray(data, mode=mode).convert("RGB")))
    cell_width = max(image.width for _, image in encoded)
    cell_height = max(image.height for _, image in encoded)
    label_height = 28
    canvas = Image.new("RGB", (cell_width * len(encoded), cell_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(encoded):
        x = index * cell_width
        canvas.paste(image, (x, label_height))
        draw.text((x + 6, 8), label, fill="black", font=font)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)

