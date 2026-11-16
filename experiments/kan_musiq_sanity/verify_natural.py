"""Verify natural-photo calibration evidence and the RGB prototype."""

import json
from pathlib import Path

import numpy as np
import torch
from models import MaskedImageKAN, checker_mask, blind_loss

ROOT = Path(__file__).resolve().parent


def main():
    rows = json.loads((ROOT / "results/natural_scores.json").read_text())["rows"]
    summary = json.loads((ROOT / "results/natural_summary.json").read_text())
    assert len(rows) == 72 and len({(r["image"], r["variant"], r["seed"]) for r in rows}) == 72
    assert all(np.isfinite(r["musiq"]) for r in rows)
    assert summary["primary_auxiliary_loss_gate"] is True
    assert all(summary["strict_monotonicity_all_levels_by_image"].values())
    for name, means in summary["scores_mean_over_noise_seeds"].items():
        for variant, expected in means.items():
            actual = np.mean([r["musiq"] for r in rows if r["image"] == name and r["variant"] == variant])
            assert abs(actual - expected) < 1e-9
    probe = json.loads((ROOT / "results/natural_quality_gradient.json").read_text())
    assert probe["metric_frozen"] and probe["input_gradient_finite"] and probe["input_gradient_norm"] > 0
    model = MaskedImageKAN(channels=3)
    checkpoint = torch.load(ROOT / "checkpoints/prototype_rgb.pt", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    sample = torch.randn(2, 3, 16, 16)
    mask = checker_mask(sample)
    a, code = model(sample, mask)
    b, other_code = model(sample + mask * 20, mask)
    assert a.shape == sample.shape and torch.equal(a, b) and torch.equal(code, other_code)
    assert torch.isfinite(blind_loss(a, sample, mask))
    for name in ("preflight_explainer.png", "natural_image_comparison.png", "natural_musiq_curve.png"):
        assert (ROOT / "results" / name).is_file()
    print("72 natural-photo scores, fixed gate, frozen quality gradients, RGB masking, and figures verified")


if __name__ == "__main__":
    main()
