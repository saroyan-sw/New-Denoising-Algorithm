"""Independent distribution, function, tail, and finite-sample evaluation."""

import csv
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import norm

from checkpoint import load_checkpoint
from models import AffineGenerator, make_kan, make_mlp, parameter_count
from state import ROOT


TEST_SIZE = 100_000
LATENT_SEED = 824_501
TARGET_SEED = 912_877
FLOOR_SEED = 725_347
QUANTILES = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]


def model_from_checkpoint(kind: str, selection: str = "best"):
    directory = ROOT / "checkpoints" if kind == "kan" else ROOT / "checkpoints" / kind
    final = load_checkpoint(directory / "latest.pt")
    checkpoint = final if selection == "final" else load_checkpoint(ROOT / final["best_checkpoint"])
    cfg = checkpoint["config"]
    if kind == "kan":
        model = make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"])
    elif kind == "mlp":
        model = make_mlp()
    else:
        model = AffineGenerator()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint, final


@torch.no_grad()
def predict(model, latent: np.ndarray) -> tuple[np.ndarray, float]:
    inputs = torch.from_numpy(latent.astype(np.float32)[:, None])
    model(inputs[:2048])  # warm the fixed CPU path before one timed inference
    begin = time.perf_counter()
    output = model(inputs).flatten().numpy().astype(np.float64)
    return output, time.perf_counter() - begin


def sample_w2(output: np.ndarray, target_sorted: np.ndarray) -> float:
    return math.sqrt(float(np.mean((np.sort(output) - target_sorted) ** 2)))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def map_diagnostics(model, std: float) -> tuple[dict, np.ndarray, np.ndarray]:
    grid = np.linspace(1e-4, 1 - 1e-4, 10_000, dtype=np.float32)
    values, _ = predict(model, grid)
    increasing = std * ndtri(grid.astype(np.float64))
    decreasing = -increasing
    rmse_inc = float(np.sqrt(np.mean((values - increasing) ** 2)))
    rmse_dec = float(np.sqrt(np.mean((values - decreasing) ** 2)))
    differences = np.diff(values)
    chosen = decreasing if rmse_dec < rmse_inc else increasing
    diagnostics = {
        "orientation": "decreasing" if rmse_dec < rmse_inc else "increasing",
        "interior_rmse_increasing": rmse_inc,
        "interior_rmse_decreasing": rmse_dec,
        "interior_rmse_chosen": min(rmse_inc, rmse_dec),
        "interior_max_absolute_error_chosen": float(np.max(np.abs(values - chosen))),
        "fraction_increasing_differences": float(np.mean(differences > 0)),
        "fraction_decreasing_differences": float(np.mean(differences < 0)),
        "sign_changes_in_differences": int(np.sum(np.sign(differences[1:]) != np.sign(differences[:-1]))),
        "rmse_left_1pct": float(np.sqrt(np.mean((values[grid <= 0.01] - chosen[grid <= 0.01]) ** 2))),
        "rmse_right_1pct": float(np.sqrt(np.mean((values[grid >= 0.99] - chosen[grid >= 0.99]) ** 2))),
    }
    return diagnostics, grid, values


def main() -> None:
    out_metrics = ROOT / "results" / "metrics"
    out_figures = ROOT / "results" / "figures"
    cfg = load_checkpoint(ROOT / "checkpoints" / "latest.pt")["config"]
    std = cfg["target_std"]
    z = np.random.default_rng(LATENT_SEED).uniform(1e-7, 1 - 1e-7, TEST_SIZE).astype(np.float32)
    target = np.random.default_rng(TARGET_SEED).normal(0, std, TEST_SIZE)
    target_sorted = np.sort(target)
    baseline = json.loads((out_metrics / "baseline_details.json").read_text(encoding="utf-8"))
    outputs = {}
    metadata = {}
    maps = {}
    for kind, label, selection in (
        ("kan", "KAN best", "best"),
        ("kan", "KAN final", "final"),
        ("mlp", "MLP best", "best"),
        ("affine", "Affine Adam best", "best"),
    ):
        model, selected, final = model_from_checkpoint(kind, selection)
        outputs[label], inference_seconds = predict(model, z)
        metadata[label] = {
            "parameters": parameter_count(model),
            "selected_step": selected["step"],
            "training_loop_seconds": final["metadata"]["elapsed_seconds"],
            "inference_seconds_100k": inference_seconds,
            "uses_gaussian_family": False,
        }
        if label in ("KAN best", "KAN final", "MLP best", "Affine Adam best"):
            maps[label] = map_diagnostics(model, std)
    outputs["Affine empirical fit"] = baseline["fitted_affine_shift"] + baseline["fitted_affine_scale"] * z
    outputs["Gaussian parameter fit"] = baseline["estimated_gaussian_mean"] + baseline["estimated_gaussian_std"] * ndtri(z.astype(np.float64))
    outputs["True inverse-CDF oracle"] = std * ndtri(z.astype(np.float64))
    for label, parameters, assumed in (
        ("Affine empirical fit", 2, False),
        ("Gaussian parameter fit", 2, True),
        ("True inverse-CDF oracle", 0, True),
    ):
        metadata[label] = {"parameters": parameters, "selected_step": "analytic", "training_loop_seconds": None, "inference_seconds_100k": None, "uses_gaussian_family": assumed}

    floor_rng = np.random.default_rng(FLOOR_SEED)
    floor_values = []
    for _ in range(10):
        a = np.sort(floor_rng.normal(0, std, TEST_SIZE))
        b = np.sort(floor_rng.normal(0, std, TEST_SIZE))
        floor_values.append(float(np.sqrt(np.mean((a - b) ** 2))))
    floor = {"mean_w2": float(np.mean(floor_values)), "sd_w2": float(np.std(floor_values, ddof=1)), "replicate_w2": floor_values}
    (out_metrics / "sampling_floor.json").write_text(json.dumps(floor, indent=2) + "\n", encoding="utf-8")

    rows = []
    quantile_rows = []
    tail_rows = []
    theoretical_quantiles = norm.ppf(QUANTILES, scale=std)
    for label, values in outputs.items():
        values = np.asarray(values, dtype=np.float64)
        outputs[label] = values
        observed_quantiles = np.quantile(values, QUANTILES)
        w2 = sample_w2(values, target_sorted)
        row = {
            "method": label,
            "test_w2": w2,
            "test_w2_squared": w2 * w2,
            "mean": float(np.mean(values)),
            "mean_error": float(np.mean(values)),
            "std": float(np.std(values)),
            "std_error": float(np.std(values) - std),
            "parameters": metadata[label]["parameters"],
            "selected_step": metadata[label]["selected_step"],
            "training_loop_seconds": metadata[label]["training_loop_seconds"],
            "inference_seconds_100k": metadata[label]["inference_seconds_100k"],
            "uses_gaussian_family": metadata[label]["uses_gaussian_family"],
        }
        rows.append(row)
        for p, observed, expected in zip(QUANTILES, observed_quantiles, theoretical_quantiles):
            quantile_rows.append({"method": label, "probability": p, "observed_quantile": float(observed), "theoretical_quantile": float(expected), "signed_error": float(observed - expected), "absolute_error": float(abs(observed - expected))})
        for threshold in (0.3, 0.4, 0.5, 0.6):
            tail_rows.append({"method": label, "absolute_threshold": threshold, "observed_fraction": float(np.mean(np.abs(values) > threshold)), "theoretical_fraction": float(2 * norm.sf(threshold / std))})
    write_csv(out_metrics / "independent_test_metrics.csv", rows)
    write_csv(out_metrics / "quantile_errors.csv", quantile_rows)
    write_csv(out_metrics / "tail_fractions.csv", tail_rows)
    map_summary = {label: item[0] for label, item in maps.items()}
    (out_metrics / "function_diagnostics.json").write_text(json.dumps(map_summary, indent=2) + "\n", encoding="utf-8")
    eval_meta = {"test_size": TEST_SIZE, "test_latent_seed": LATENT_SEED, "test_target_seed": TARGET_SEED, "floor_seed": FLOOR_SEED, "floor_replicates": 10, "interior_z_interval": [1e-4, 1 - 1e-4], "checkpoint_selection": "best validation checkpoint, before independent test", "no_test_based_model_selection": True}
    (out_metrics / "evaluation_protocol.json").write_text(json.dumps(eval_meta, indent=2) + "\n", encoding="utf-8")

    # Histograms and QQ use the same untouched independent test arrays.
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    edges = np.linspace(-0.8, 0.8, 100)
    ax.hist(target, bins=edges, density=True, histtype="step", lw=2, label="Independent target")
    for label in ("KAN best", "MLP best", "Affine Adam best", "Gaussian parameter fit"):
        ax.hist(outputs[label], bins=edges, density=True, histtype="step", lw=1.3, label=label)
    ax.set(xlabel="Value", ylabel="Density", title="Independent 100,000-sample output laws")
    ax.legend()
    fig.savefig(out_figures / "independent_histograms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    probs = np.linspace(0.001, 0.999, 500)
    ref = np.quantile(target, probs)
    for label in ("KAN best", "KAN final", "MLP best", "Affine Adam best", "Gaussian parameter fit"):
        ax.plot(ref, np.quantile(outputs[label], probs), label=label, lw=1.3)
    ax.plot([-0.7, 0.7], [-0.7, 0.7], "k--", label="Perfect agreement")
    ax.set(xlabel="Independent target quantiles", ylabel="Generated quantiles", title="Quantile–quantile comparison", xlim=(-0.7, 0.7), ylim=(-0.7, 0.7))
    ax.legend(fontsize=8)
    fig.savefig(out_figures / "independent_qq.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for kind, label in (("kan_primary_seed_1729", "KAN"), ("mlp_seed_1729", "MLP"), ("affine_seed_1729", "Affine Adam")):
        with (out_metrics / f"{kind}.csv").open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        ax.plot([int(row["step"]) for row in history], [math.sqrt(float(row["val_w2_squared"])) for row in history], label=f"{label} validation")
        trained = [row for row in history if row["train_w2_squared"]]
        ax.plot([int(row["step"]) for row in trained], [math.sqrt(float(row["train_w2_squared"])) for row in trained], ls="--", alpha=0.55, label=f"{label} train")
    ax.set(yscale="log", xlabel="Optimizer step", ylabel="Empirical W2", title="Training and validation progress")
    ax.legend()
    fig.savefig(out_figures / "validation_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    grid = maps["KAN best"][1]
    ax.plot(grid, std * ndtri(grid.astype(np.float64)), "k--", label="Increasing inverse CDF")
    ax.plot(grid, -std * ndtri(grid.astype(np.float64)), "k:", label="Decreasing inverse CDF")
    for label in ("KAN best", "KAN final", "MLP best", "Affine Adam best"):
        ax.plot(maps[label][1], maps[label][2], label=label, lw=1.4)
    ax.set(xlabel="Uniform input z", ylabel="Output", title="Learned maps on [1e-4, 1-1e-4]", ylim=(-0.8, 0.8))
    ax.legend(fontsize=8)
    fig.savefig(out_figures / "inverse_cdf_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    p_tail = [0.001, 0.01, 0.99, 0.999]
    expected = norm.ppf(p_tail, scale=std)
    for label in ("KAN best", "KAN final", "MLP best", "Affine Adam best", "Gaussian parameter fit"):
        ax.plot(range(4), np.quantile(outputs[label], p_tail) - expected, marker="o", label=label)
    ax.axhline(0, color="black", ls="--")
    ax.set(xticks=range(4), xticklabels=[str(p) for p in p_tail], xlabel="Quantile probability", ylabel="Signed error versus Gaussian", title="Tail quantile errors")
    ax.legend(fontsize=8)
    fig.savefig(out_figures / "tail_quantile_errors.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    for ax, name in zip(axes, ("initial.pt", "step_000100.pt", "best_step_000400.pt", "latest.pt")):
        checkpoint = load_checkpoint(ROOT / "checkpoints" / name)
        cfg = checkpoint["config"]
        model = make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"])
        model.load_state_dict(checkpoint["model_state"])
        values, _ = predict(model, z[:20_000])
        ax.hist(target[:20_000], bins=edges, density=True, alpha=0.4, label="Target")
        ax.hist(values, bins=edges, density=True, alpha=0.5, label="KAN")
        ax.set(title=f"Step {checkpoint['step']}", xlabel="Output", xlim=(-0.8, 0.8))
    axes[0].legend(fontsize=8)
    fig.savefig(out_figures / "distribution_snapshots.png", dpi=150)
    plt.close(fig)
    for row in rows:
        print(f"{row['method']}: independent W2={row['test_w2']:.6f}, std={row['std']:.6f}")
    print(f"Gaussian-vs-Gaussian sampling floor: mean W2={floor['mean_w2']:.6f} (sd {floor['sd_w2']:.6f})")


if __name__ == "__main__":
    main()
