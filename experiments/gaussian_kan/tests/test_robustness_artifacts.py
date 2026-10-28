"""Validate the complete robustness queue and confirmation comparison."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from checkpoint import load_checkpoint
from experiment_runner import ExperimentConfig, build_model
from train import fixed_validation, validate


def main():
    plan = json.loads((ROOT / "config/robustness.json").read_text())
    with (ROOT / "results/metrics/robustness_comparison.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    summary = json.loads((ROOT / "results/metrics/robustness_summary.json").read_text())
    assert abs(np.mean([float(row["confirmation_w2"]) for row in rows[:3]]) - summary["primary_three_seed_mean_w2"]) < 1e-12
    for item in plan["runs"]:
        directory = ROOT / "runs" / item["run_id"]
        state = json.loads((directory / "state.json").read_text())
        assert state["status"] == "completed" and state["checkpoint_step"] == 1000
        assert len(list((directory / "checkpoints").glob("step_*.pt"))) == 10
        final = load_checkpoint(directory / "checkpoints/latest.pt")
        best = load_checkpoint(directory / "checkpoints" / final["best_checkpoint"])
        config = ExperimentConfig(**final["config"])
        model = build_model(config)
        model.load_state_dict(best["model_state"])
        actual = validate(model, *fixed_validation(config.validation_size, config.target_std, "cpu"))
        assert abs(actual["val_w2_squared"] - best["validation"]["val_w2_squared"]) < 1e-9
    assert [int(row["parameters"]) for row in rows] == [2960, 2960, 2960, 592, 5520]
    print("All four robustness runs, 40 periodic checkpoints, and confirmation summary verified")


if __name__ == "__main__":
    main()
