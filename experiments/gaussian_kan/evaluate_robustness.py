"""Evaluate the fixed seed/capacity queue on a fresh confirmation pair."""

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtri

from checkpoint import load_checkpoint
from evaluate import map_diagnostics, model_from_checkpoint, predict, sample_w2, write_csv
from experiment_runner import ExperimentConfig, build_model
from state import ROOT


def main():
    plan = json.loads((ROOT / "config/robustness.json").read_text())
    confirmation = plan["confirmation"]
    z = np.random.default_rng(confirmation["latent_seed"]).uniform(1e-7, 1 - 1e-7, confirmation["size"]).astype(np.float32)
    target = np.sort(np.random.default_rng(confirmation["target_seed"]).normal(0, 0.2, confirmation["size"]))
    primary, best, final = model_from_checkpoint("kan")
    records = [("kan_primary_seed_1729", primary, best, final, 108.89759960002266)]
    for item in plan["runs"]:
        directory = ROOT / "runs" / item["run_id"]
        state = json.loads((directory / "state.json").read_text())
        assert state["status"] == "completed", f"unfinished run {item['run_id']}"
        final = load_checkpoint(directory / "checkpoints/latest.pt")
        assert final["step"] == 1000
        assert len(list((directory / "checkpoints").glob("step_*.pt"))) == 10
        best = load_checkpoint(directory / "checkpoints" / final["best_checkpoint"])
        config = ExperimentConfig(**final["config"])
        model = build_model(config)
        model.load_state_dict(best["model_state"])
        records.append((item["run_id"], model, best, final, final["metadata"]["elapsed_seconds"]))
    rows, maps = [], {}
    for name, model, best, final, seconds in records:
        values, inference_seconds = predict(model, z)
        diagnostic, grid, curve = map_diagnostics(model, 0.2)
        maps[name] = (grid, curve, diagnostic)
        model.load_state_dict(final["model_state"])
        final_values, _ = predict(model, z)
        q = np.quantile(values, [0.001, 0.999])
        expected = 0.2 * ndtri([0.001, 0.999])
        rows.append({
            "run_id": name, "widths": str(best["config"]["widths"]),
            "grid_intervals": best["config"]["grid_intervals"], "seed": best["config"]["seed"],
            "parameters": best["metadata"]["parameter_count"], "selected_step": best["step"],
            "best_val_w2": math.sqrt(best["validation"]["val_w2_squared"]),
            "confirmation_w2": sample_w2(values, target), "final_confirmation_w2": sample_w2(final_values, target),
            "output_mean": float(values.mean()), "output_std": float(values.std()),
            "interior_map_rmse": diagnostic["interior_rmse_chosen"], "orientation": diagnostic["orientation"],
            "fraction_increasing": diagnostic["fraction_increasing_differences"],
            "fraction_decreasing": diagnostic["fraction_decreasing_differences"],
            "q001_error": float(q[0] - expected[0]), "q999_error": float(q[1] - expected[1]),
            "fraction_abs_above_06": float(np.mean(np.abs(values) > 0.6)),
            "training_loop_seconds": seconds, "inference_seconds_100k": inference_seconds,
        })
    floor_rng = np.random.default_rng(20261031)
    floor = [float(np.sqrt(np.mean((np.sort(floor_rng.normal(0, .2, confirmation["size"])) - np.sort(floor_rng.normal(0, .2, confirmation["size"]))) ** 2))) for _ in range(10)]
    values = [row["confirmation_w2"] for row in rows[:3]]
    summary = {
        "protocol": confirmation, "sampling_floor_seed": 20261031, "floor_replicates": floor,
        "floor_mean_w2": float(np.mean(floor)), "floor_sd_w2": float(np.std(floor, ddof=1)),
        "oracle_confirmation_w2": sample_w2(0.2 * ndtri(z.astype(np.float64)), target),
        "primary_three_seed_mean_w2": float(np.mean(values)), "primary_three_seed_sd_w2": float(np.std(values, ddof=1)),
        "primary_three_seed_range_w2": [float(min(values)), float(max(values))],
        "primary_orientations": [row["orientation"] for row in rows[:3]],
        "checkpoint_selection": "best fixed-validation checkpoint for each run; final checkpoints reported separately",
        "interpretation_limit": "Three primary seeds; capacity alternatives each have one seed. Exploratory after first independent evaluation. No claim of minimal possible capacity or general KAN-versus-MLP superiority."
    }
    write_csv(ROOT / "results/metrics/robustness_comparison.csv", rows)
    (ROOT / "results/metrics/robustness_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    labels = ["Primary\n1729", "Primary\n1730", "Primary\n1731", "Small 32\n1729", "Primary 64\n1729"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    positions = np.arange(len(rows))
    axes[0].bar(positions - .18, [r["confirmation_w2"] for r in rows], width=.36, label="Validation-selected")
    axes[0].bar(positions + .18, [r["final_confirmation_w2"] for r in rows], width=.36, label="Final step")
    axes[0].axhline(summary["floor_mean_w2"], color="black", ls="--", label="Sampling reference mean")
    axes[0].set(xticks=positions, xticklabels=labels, ylabel="Confirmation W2", title="Seed and capacity comparison")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [r["parameters"] for r in rows])
    axes[1].set(ylabel="Trainable parameters", title="Spline capacity")
    fig.savefig(ROOT / "results/figures/robustness_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for name, (grid, curve, diagnostic) in maps.items():
        aligned = curve[::-1] if diagnostic["orientation"] == "decreasing" else curve
        reference = .2 * ndtri(grid.astype(np.float64))
        axes[0].plot(grid, aligned, lw=1.2, label=name.replace("kan_", ""))
        axes[1].plot(grid, aligned - reference, lw=1.2, label=name.replace("kan_", ""))
    axes[0].plot(grid, reference, "k--", lw=1.5, label="Increasing oracle")
    axes[0].set(xlabel="z (orientation aligned for display)", ylabel="Output", title="Learned maps", ylim=(-.8, .8))
    axes[1].axhline(0, color="black", ls="--")
    axes[1].set(xlabel="z (orientation aligned for display)", ylabel="Error versus inverse CDF", title="Interior and tail approximation error")
    axes[0].legend(fontsize=7)
    fig.savefig(ROOT / "results/figures/capacity_maps.png", dpi=150)
    plt.close(fig)
    for row in rows:
        print(f"{row['run_id']}: W2={row['confirmation_w2']:.6f}, params={row['parameters']}, orientation={row['orientation']}")
    print(f"Primary three-seed W2 mean={summary['primary_three_seed_mean_w2']:.6f}, SD={summary['primary_three_seed_sd_w2']:.6f}")


if __name__ == "__main__":
    main()
