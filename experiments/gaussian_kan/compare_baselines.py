"""Compare neural, empirical affine, Gaussian-fit, and oracle generators."""

import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import ndtri

from checkpoint import load_checkpoint
from models import AffineGenerator, make_kan, make_mlp, parameter_count
from state import ROOT, load_state, update_state
from train import fixed_validation


def measured_row(name, generated, target, count, train_seconds, best_step, assumption, training_samples):
    output = np.asarray(generated, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    w2sq = float(np.mean((np.sort(output) - np.sort(truth)) ** 2))
    return {
        "method": name,
        "trainable_parameters": count,
        "best_step": best_step,
        "training_samples": training_samples,
        "training_loop_seconds": train_seconds,
        "val_w2_squared": w2sq,
        "val_w2": math.sqrt(w2sq),
        "output_mean": float(output.mean()),
        "output_std": float(output.std()),
        "assumption": assumption,
    }


def neural_prediction(kind: str, latent: torch.Tensor):
    directory = ROOT / "checkpoints" / kind if kind != "kan" else ROOT / "checkpoints"
    final = load_checkpoint(directory / "latest.pt")
    best = load_checkpoint(ROOT / final["best_checkpoint"])
    cfg = best["config"]
    model = {
        "kan": lambda: make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"]),
        "mlp": make_mlp,
        "affine": AffineGenerator,
    }[kind]()
    model.load_state_dict(best["model_state"])
    model.eval()
    with torch.no_grad():
        generated = model(latent).cpu().numpy().reshape(-1)
    return generated, parameter_count(model), best["step"], final["metadata"]["elapsed_seconds"]


def main() -> None:
    cfg = load_checkpoint(ROOT / "checkpoints" / "latest.pt")["config"]
    latent, target = fixed_validation(cfg["validation_size"], cfg["target_std"], "cpu")
    z = latent.numpy().reshape(-1)
    truth = target.numpy().reshape(-1)
    rows = []
    neural_outputs = {}
    for kind, label in (("kan", "KAN best"), ("mlp", "MLP best"), ("affine", "Affine Adam best")):
        generated, count, best_step, seconds = neural_prediction(kind, latent)
        neural_outputs[kind] = generated
        rows.append(measured_row(label, generated, truth, count, seconds, best_step, "distribution-free", cfg["batch_size"] * cfg["steps"]))

    # Separate independent unpaired training sets; same target draw budget as each neural run.
    sample_count = cfg["batch_size"] * cfg["steps"]
    rng_z = np.random.default_rng(33991)
    rng_n = np.random.default_rng(44117)
    begin = time.perf_counter()
    fit_z = np.sort(rng_z.uniform(1e-7, 1 - 1e-7, sample_count))
    fit_n = np.sort(rng_n.normal(0, cfg["target_std"], sample_count))
    z_mean, n_mean = float(fit_z.mean()), float(fit_n.mean())
    slope = float(np.mean((fit_z - z_mean) * (fit_n - n_mean)) / np.var(fit_z))
    intercept = n_mean - slope * z_mean
    affine_fit_seconds = time.perf_counter() - begin
    rows.append(measured_row("Affine empirical fit", intercept + slope * z, truth, 2, affine_fit_seconds, "closed form", "distribution-free sorted quantile fit", sample_count))

    # Uses the Gaussian family assumption; not a distribution-free neural baseline.
    begin = time.perf_counter()
    gaussian_fit_samples = np.random.default_rng(44117).normal(0, cfg["target_std"], sample_count)
    estimated_mean = float(gaussian_fit_samples.mean())
    estimated_std = float(gaussian_fit_samples.std(ddof=0))
    gaussian_fit_seconds = time.perf_counter() - begin
    q = ndtri(z.astype(np.float64))
    rows.append(measured_row("Gaussian parameter fit", estimated_mean + estimated_std * q, truth, 2, gaussian_fit_seconds, "closed form", "Gaussian family and inverse CDF", sample_count))
    rows.append(measured_row("True inverse-CDF oracle", cfg["target_std"] * q, truth, 0, 0.0, "oracle", "true target parameters; evaluation only", 0))

    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "baseline_comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    details = {
        "validation_size": cfg["validation_size"],
        "latent_seed": 917271,
        "target_seed": 638219,
        "fit_latent_seed": 33991,
        "fit_target_seed": 44117,
        "fitted_affine_scale": slope,
        "fitted_affine_shift": intercept,
        "estimated_gaussian_mean": estimated_mean,
        "estimated_gaussian_std": estimated_std,
        "comparison_limit": "Best neural checkpoints were selected on this fixed validation set; independent held-out evaluation follows in Milestone 4.",
    }
    (out_dir / "baseline_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    labels = [row["method"].replace(" ", "\n") for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(labels, [row["val_w2"] for row in rows])
    axes[0].set(ylabel="Fixed-validation W2 (lower is better)", title="Distribution match")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar(labels[:-1], [float(row["training_loop_seconds"]) for row in rows[:-1]])
    axes[1].set(yscale="log", ylabel="Fit/training seconds (log scale)", title="Measured fit cost")
    axes[1].tick_params(axis="x", labelsize=8)
    figure = ROOT / "results" / "figures" / "baseline_comparison.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    edges = np.linspace(-0.7, 0.7, 80)
    axes[0].hist(truth, bins=edges, density=True, histtype="step", lw=2, label="Target")
    for kind, label in (("kan", "KAN"), ("mlp", "MLP"), ("affine", "Affine Adam")):
        axes[0].hist(neural_outputs[kind], bins=edges, density=True, histtype="step", lw=1.5, label=label)
    axes[0].set(xlabel="Output value", ylabel="Density", title="Fixed validation distributions")
    axes[0].legend()
    grid = np.linspace(1e-4, 1 - 1e-4, 1000, dtype=np.float32)
    grid_tensor = torch.from_numpy(grid[:, None])
    axes[1].plot(grid, cfg["target_std"] * ndtri(grid.astype(np.float64)), "k--", lw=1.5, label="Increasing oracle")
    axes[1].plot(grid, -cfg["target_std"] * ndtri(grid.astype(np.float64)), "k:", lw=1.5, label="Decreasing oracle")
    for kind, label in (("kan", "KAN"), ("mlp", "MLP"), ("affine", "Affine Adam")):
        curve, _, _, _ = neural_prediction(kind, grid_tensor)
        axes[1].plot(grid, curve, lw=1.5, label=label)
    axes[1].plot(grid, intercept + slope * grid, lw=1.2, label="Affine empirical fit")
    axes[1].set(xlabel="Uniform input z", ylabel="Output", title="Learned transformations", ylim=(-0.7, 0.7))
    axes[1].legend(fontsize=8)
    fig.savefig(ROOT / "results" / "figures" / "baseline_maps.png", dpi=150)
    plt.close(fig)
    completed = list(load_state()["completed_experiments"])
    if "gaussian_parameter_estimate" not in completed:
        completed.append("gaussian_parameter_estimate")
    pending = [x for x in load_state()["pending_experiments"] if x != "gaussian_parameter_estimate"]
    update_state(completed_experiments=completed, pending_experiments=pending, status="completed", next_action="Verify baseline table and proceed to independent scientific evaluation")
    for row in rows:
        print(f"{row['method']}: W2={row['val_w2']:.6f}, seconds={row['training_loop_seconds']:.3f}")
    print(table)


if __name__ == "__main__":
    main()
