"""Verify saved baseline checkpoints and the shared validation comparison."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_checkpoint
from models import AffineGenerator, make_mlp, parameter_count
from train import fixed_validation, validate


def main() -> None:
    latent, target = fixed_validation(8192, 0.2, "cpu")
    with (ROOT / "results" / "metrics" / "baseline_comparison.csv").open(newline="", encoding="utf-8") as handle:
        table = {row["method"]: row for row in csv.DictReader(handle)}
    for kind, label, factory, count in (
        ("mlp", "MLP best", make_mlp, 2913),
        ("affine", "Affine Adam best", AffineGenerator, 2),
    ):
        directory = ROOT / "checkpoints" / kind
        final = load_checkpoint(directory / "latest.pt")
        best = load_checkpoint(ROOT / final["best_checkpoint"])
        assert final["step"] == 1000 and parameter_count(factory()) == count
        assert len(list(directory.glob("step_*.pt"))) == 10
        model = factory()
        model.load_state_dict(best["model_state"])
        recomputed = validate(model, latent, target)
        assert abs(recomputed["val_w2_squared"] - float(table[label]["val_w2_squared"])) < 1e-8
        assert abs(recomputed["val_w2_squared"] - best["validation"]["val_w2_squared"]) < 1e-8
    details = json.loads((ROOT / "results" / "metrics" / "baseline_details.json").read_text())
    assert 0.6 < details["fitted_affine_scale"] < 0.75
    assert abs(details["estimated_gaussian_std"] - 0.2) < 0.001
    print("Baseline checkpoints and shared validation metrics verified")


if __name__ == "__main__":
    main()
