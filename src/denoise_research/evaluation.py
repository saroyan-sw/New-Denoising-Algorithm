from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .degradations import standard_suite
from .io import load_image
from .metrics import full_reference_metrics
from .pipelines import PaperInspiredPipeline


def benchmark_images(paths: Iterable[Path], seed: int = 0) -> list[dict[str, object]]:
    pipeline = PaperInspiredPipeline()
    rows: list[dict[str, object]] = []
    for image_index, path in enumerate(paths):
        clean = load_image(path)
        for degradation, distorted in standard_suite(clean, seed + image_index).items():
            restored = pipeline.run(distorted)
            for method, candidate in (("distorted", distorted), ("paper_inspired", restored)):
                row: dict[str, object] = {
                    "image": str(path),
                    "degradation": degradation,
                    "method": method,
                    "seed": seed + image_index,
                }
                row.update(full_reference_metrics(clean, candidate))
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows to write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

