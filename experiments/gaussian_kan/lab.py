"""Small public helpers for notebook inspection and controlled comparisons."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.special import ndtri

from checkpoint import load_checkpoint
from experiment_runner import ExperimentConfig, build_model
from models import AffineGenerator, make_kan, make_mlp, parameter_count
from state import ROOT


@dataclass
class GeneratorArtifact:
    name: str
    predict_function: Callable
    parameters: int
    provenance: dict

    def predict(self, latent):
        return np.asarray(self.predict_function(np.asarray(latent, dtype=np.float32)), dtype=np.float64).reshape(-1)


def _model_predictor(model):
    model.eval()
    def predict(latent):
        with torch.no_grad():
            return model(torch.from_numpy(latent.reshape(-1, 1))).cpu().numpy().reshape(-1)
    return predict


def load_artifact(name, root=ROOT):
    """Original aliases or a named run under runs/ or explorations/."""
    root = Path(root)
    original = {"kan_best": ("kan", "best"), "kan_final": ("kan", "final"),
                "mlp_best": ("mlp", "best"), "affine_best": ("affine", "best")}
    if name in original:
        kind, selection = original[name]
        directory = root / "checkpoints" if kind == "kan" else root / "checkpoints" / kind
        final = load_checkpoint(directory / "latest.pt")
        reference = str(final["best_checkpoint"]).replace("\\", "/")
        selected = final if selection == "final" else load_checkpoint(root / reference)
        cfg = selected["config"]
        model = make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"]) if kind == "kan" else make_mlp() if kind == "mlp" else AffineGenerator()
        model.load_state_dict(selected["model_state"])
        return GeneratorArtifact(name, _model_predictor(model), parameter_count(model),
                                 {"step": selected["step"], "config": cfg, "selection": selection, "uses_gaussian_family": False})
    if name in {"affine_empirical", "gaussian_fit", "oracle"}:
        fit = json.loads((root / "results/metrics/baseline_details.json").read_text())
        if name == "affine_empirical":
            predictor = lambda z: fit["fitted_affine_shift"] + fit["fitted_affine_scale"] * z.astype(np.float64)
        elif name == "gaussian_fit":
            predictor = lambda z: fit["estimated_gaussian_mean"] + fit["estimated_gaussian_std"] * ndtri(z.astype(np.float64))
        else:
            predictor = lambda z: 0.2 * ndtri(z.astype(np.float64))
        return GeneratorArtifact(name, predictor, 0 if name == "oracle" else 2,
                                 {"uses_gaussian_family": name != "affine_empirical", "selection": "analytical comparator"})
    for parent in (root / "runs", root / "explorations"):
        directory = parent / name
        if not (directory / "config.json").exists():
            continue
        cfg = ExperimentConfig(**json.loads((directory / "config.json").read_text()))
        final = load_checkpoint(directory / "checkpoints/latest.pt")
        selected = load_checkpoint(directory / "checkpoints" / final["best_checkpoint"])
        model = build_model(cfg)
        model.load_state_dict(selected["model_state"])
        state = json.loads((directory / "state.json").read_text())
        return GeneratorArtifact(name, _model_predictor(model), parameter_count(model),
                                 {"step": selected["step"], "config": selected["config"], "status": state["status"], "uses_gaussian_family": False})
    raise FileNotFoundError(f"No saved artifact {name!r}; inspect the available checkpoints or run it first")


def compare_artifacts(names, *, sample_size=20_000, seed=7101, target_std=0.2, root=ROOT):
    """Fresh independent samples; comparison never chooses or changes weights."""
    latent = np.random.default_rng(seed).uniform(1e-7, 1 - 1e-7, sample_size).astype(np.float32)
    target = np.random.default_rng(seed + 1).normal(0, target_std, sample_size)
    sorted_target = np.sort(target)
    rows, outputs = [], {}
    for name in names:
        artifact = load_artifact(name, root)
        trained_std = artifact.provenance.get("config", {}).get("target_std", 0.2)
        if not np.isclose(trained_std, target_std):
            raise ValueError(f"{name} was trained for std={trained_std}; use comparators for the same target law")
        values = artifact.predict(latent)
        outputs[name] = values
        rows.append({"artifact": name, "W2": float(np.sqrt(np.mean((np.sort(values) - sorted_target) ** 2))),
                     "mean_error": float(values.mean()), "std_error": float(values.std() - target_std),
                     "parameters": artifact.parameters, "Gaussian_assumption": artifact.provenance.get("uses_gaussian_family", False),
                     "status": artifact.provenance.get("status", "completed/reference")})
    return rows, latent, target, outputs
