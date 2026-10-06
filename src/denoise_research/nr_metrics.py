from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class OptionalMetricDependencyError(RuntimeError):
    pass


def _dependencies() -> tuple[Any, Any]:
    try:
        import pyiqa
        import torch
    except ImportError as error:
        raise OptionalMetricDependencyError(
            "PyIQA metrics require the optional 'torch' dependencies. "
            "Install this project with: pip install -e .[torch]"
        ) from error
    return pyiqa, torch


def _tensor(image: np.ndarray, torch: Any, device: Any) -> Any:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected HxW or HxWx3 image")
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).clamp(0, 1)


class PyIQAMetrics:
    """Lazy wrapper around the same metric family used by the archived project."""

    def __init__(self, device: str | None = None):
        pyiqa, torch = _dependencies()
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.brisque_metric = pyiqa.create_metric("brisque").to(self.device).eval()
        self.niqe_metric = pyiqa.create_metric("niqe").to(self.device).eval()
        self.vif_metric = pyiqa.create_metric("vif").to(self.device).eval()

    def brisque(self, image: np.ndarray) -> float:
        with self.torch.no_grad():
            return float(self.brisque_metric(_tensor(image, self.torch, self.device)).item())

    def niqe(self, image: np.ndarray) -> float:
        with self.torch.no_grad():
            return float(self.niqe_metric(_tensor(image, self.torch, self.device)).item())

    def vif(self, reference: np.ndarray, estimate: np.ndarray) -> float:
        with self.torch.no_grad():
            ref = _tensor(reference, self.torch, self.device)
            est = _tensor(estimate, self.torch, self.device)
            return float(self.vif_metric(est, ref).item())


@dataclass(frozen=True)
class MetricSpec:
    """No-reference metric name and the direction representing better quality."""

    name: str
    lower_is_better: bool


def parse_metric_spec(value: str) -> MetricSpec:
    """Parse ``metric:lower`` or ``metric:higher`` from a command-line value."""
    try:
        name, direction = (part.strip() for part in value.rsplit(":", 1))
    except ValueError as error:
        raise ValueError("Metric must be written as NAME:lower or NAME:higher") from error
    if not name or direction not in {"lower", "higher"}:
        raise ValueError("Metric must be written as NAME:lower or NAME:higher")
    return MetricSpec(name=name, lower_is_better=direction == "lower")


class PyIQACollection:
    """Load only the PyIQA no-reference metrics requested by an experiment."""

    def __init__(self, specs: Sequence[MetricSpec], device: str | None = None):
        if not specs:
            raise ValueError("At least one metric is required")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique")
        pyiqa, torch = _dependencies()
        self.specs = tuple(specs)
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.metrics = {
            spec.name: pyiqa.create_metric(spec.name).to(self.device).eval()
            for spec in self.specs
        }

    def score(self, image: np.ndarray) -> dict[str, float]:
        value = _tensor(image, self.torch, self.device)
        with self.torch.no_grad():
            return {name: float(metric(value).item()) for name, metric in self.metrics.items()}
