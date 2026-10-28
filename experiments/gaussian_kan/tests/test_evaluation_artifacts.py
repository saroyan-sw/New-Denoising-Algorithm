"""Recompute held-out KAN metrics and validate the evaluation evidence."""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from evaluate import LATENT_SEED, TARGET_SEED, TEST_SIZE, model_from_checkpoint, predict, sample_w2


def main():
    metrics = ROOT / "results" / "metrics"
    with (metrics / "independent_test_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert len(rows) == 7
    latent = np.random.default_rng(LATENT_SEED).uniform(1e-7, 1 - 1e-7, TEST_SIZE).astype(np.float32)
    target = np.random.default_rng(TARGET_SEED).normal(0, 0.2, TEST_SIZE)
    model, _, _ = model_from_checkpoint("kan", "best")
    output, _ = predict(model, latent)
    expected = rows["KAN best"]
    assert abs(sample_w2(output, np.sort(target)) - float(expected["test_w2"])) < 1e-9
    for row in rows.values():
        assert abs(float(row["test_w2_squared"]) - float(row["test_w2"]) ** 2) < 1e-12
        assert abs(float(row["std_error"]) - (float(row["std"]) - 0.2)) < 1e-12
    floor = json.loads((metrics / "sampling_floor.json").read_text())
    assert len(floor["replicate_w2"]) == 10
    assert math.isclose(np.mean(floor["replicate_w2"]), floor["mean_w2"])
    diagnostic = json.loads((metrics / "function_diagnostics.json").read_text())["KAN best"]
    assert diagnostic["orientation"] == "decreasing"
    assert diagnostic["fraction_decreasing_differences"] == 1.0
    assert len(list(csv.DictReader((metrics / "quantile_errors.csv").open()))) == 49
    assert len(list(csv.DictReader((metrics / "tail_fractions.csv").open()))) == 28
    for name in ["independent_histograms", "validation_curves", "inverse_cdf_comparison", "independent_qq", "tail_quantile_errors", "distribution_snapshots"]:
        assert (ROOT / "results" / "figures" / f"{name}.png").is_file()
    print("Independent metrics, quantiles, sampling reference, and six figures verified")


if __name__ == "__main__":
    main()
