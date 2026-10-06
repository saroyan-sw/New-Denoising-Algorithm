from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperInspiredConfig:
    bilateral_radius: int = 4
    bilateral_sigma_spatial: float = 3.0
    bilateral_sigma_range: float = 0.10
    edge_threshold_std: float = 3.0
    edge_strength: float = 0.20
    gabor_kernel_size: int = 9
    gabor_sigma: float = 2.0
    gabor_wavelength: float = 4.0
    gabor_aspect_ratio: float = 0.5
    gabor_orientations: int = 8
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"low": 0.45, "enhanced": 0.45, "texture": 0.10}
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "PaperInspiredConfig":
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(frozen=True)
class ComplementaryBandConfig:
    fine_sigma: float = 1.0
    coarse_sigma: float = 2.0
    weights: dict[str, float] = field(
        default_factory=lambda: {"low": 1.0, "mid": 1.0, "high": 1.0}
    )

    @classmethod
    def from_json(cls, path: str | Path) -> "ComplementaryBandConfig":
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("optimization", None)
        return cls(**data)

