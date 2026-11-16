"""Bounded mechanical checks; no full hybrid training or clean-target training."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from models import MaskedImageKAN, checker_mask, blind_loss


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, path)


def patterns():
    yy, xx = np.mgrid[:64, :64]
    return {"black": np.zeros((64, 64)), "gray": np.full((64, 64), .5),
            "step": np.where(xx < 32, .3, .7), "blocks": .3 + .4 * ((xx // 8 + yy // 8) % 2)}


def mathematical_controls():
    rng = np.random.default_rng(20261116)
    rows = []
    for name, clean in patterns().items():
        noise = rng.normal(0, .1, (128, 64, 64))
        noisy = clean[None] + noise  # signed floats, never clipped
        baseline_mse = float(np.mean(noise ** 2))
        total_eta = np.zeros_like(noise)
        total_individual_mse = 0.
        for k in range(1, 65):
            independent_eta = rng.normal(0, .1, noise.shape)
            total_eta += independent_eta
            total_individual_mse += float(np.mean((noisy - independent_eta - clean) ** 2))
            if k in (1, 8, 64):
                averaged = noisy - total_eta / k
                rows.append({"pattern": name, "candidates": k, "noisy_mse": baseline_mse,
                             "individual_subtraction_mse": total_individual_mse / k,
                             "averaged_subtraction_mse": float(np.mean((averaged - clean) ** 2)),
                             "theoretical_average_mse": .01 * (1 + 1 / k)})
        # A fixed linear predictor excluding the center is a clean-free control.
        tensor = torch.tensor(noisy, dtype=torch.float32)[:, None]
        kernel = torch.tensor([[0., .25, 0.], [.25, 0., .25], [0., .25, 0.]])[None, None]
        estimate = F.conv2d(F.pad(tensor, (1, 1, 1, 1), mode="reflect"), kernel).numpy()[:, 0]
        noisy_risk = float(np.mean((estimate - noisy) ** 2))
        clean_risk = float(np.mean((estimate - clean) ** 2))
        rows.append({"pattern": name, "method": "center-excluding neighbor mean",
                     "self_supervised_risk": noisy_risk, "evaluation_clean_mse": clean_risk,
                     "difference_expected_sigma_squared": noisy_risk - clean_risk})
    assert all(row["averaged_subtraction_mse"] > row["noisy_mse"] for row in rows if "candidates" in row)
    amplitude = [{"amplitude": a, "expected_loss_to_noisy_input": .01 * a * a} for a in (0, .25, .5, 1.)]
    write_json({"sigma_control": .1, "replicates": 128, "rows": rows, "amplitude_control": amplitude,
                "clean_references_used_for_training": False,
                "interpretation": "Even oracle independent noise samples worsen MSE. Matching the noisy input favors zero subtraction."},
               ROOT / "results/mechanical_controls.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    values = [r for r in rows if r.get("pattern") == "step" and "candidates" in r]
    axes[0].plot([r["candidates"] for r in values], [r["averaged_subtraction_mse"] for r in values], "o-", label="Measured")
    axes[0].axhline(.01, color="black", ls="--", label="Noisy input MSE")
    axes[0].set(xlabel="Independent candidates averaged", ylabel="Clean-evaluation MSE", title="Random subtraction control")
    axes[0].legend()
    axes[1].plot([r["amplitude"] for r in amplitude], [r["expected_loss_to_noisy_input"] for r in amplitude], "o-")
    axes[1].set(xlabel="Generated-noise amplitude", ylabel="Loss to noisy input", title="The proposed fidelity loss favors zero noise")
    fig.savefig(ROOT / "results/mechanical_controls.png", dpi=150)
    plt.close(fig)


def prototype_checks(condition_noise=False):
    random.seed(71); np.random.seed(71); torch.manual_seed(71)
    model = MaskedImageKAN(condition_noise=condition_noise)
    noisy = torch.randn(2, 1, 32, 32) * .1 + .5
    mask = checker_mask(noisy)
    prediction, context = model(noisy, mask)
    altered_prediction, altered_context = model(noisy + 100 * mask, mask)
    assert torch.equal(prediction, altered_prediction) and torch.equal(context, altered_context)
    image_optimizer = torch.optim.Adam([*model.encoder.parameters(), *model.context.parameters(), *model.decoder.parameters()], lr=.001)
    before = model.decoder[-1].weight.detach().clone()
    loss = blind_loss(prediction, noisy, mask)
    image_optimizer.zero_grad(); loss.backward(); image_optimizer.step()
    assert not torch.equal(before, model.decoder[-1].weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for part in (model.encoder, model.context, model.decoder) for p in part.parameters())
    # Residuals are noisy observations minus a blind prediction, NOT true noise.
    with torch.no_grad():
        prediction, context = model(noisy, mask)
        residuals = torch.stack([(noisy - prediction)[b][mask[b].bool()][:256] for b in range(2)])
    prior_optimizer = torch.optim.Adam(model.noise_prior.parameters(), lr=.001)
    generated = model.sample_noise(torch.rand(2, 256, 1), context.detach())
    prior_loss = (generated.sort(dim=1).values - residuals.sort(dim=1).values).square().mean()
    prior_optimizer.zero_grad(); prior_loss.backward(); prior_optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.noise_prior.parameters())
    directory = ROOT / "checkpoints"
    directory.mkdir(exist_ok=True)
    checkpoint_name = "prototype_conditional" if condition_noise else "prototype_shared"
    temporary = directory / f"{checkpoint_name}.tmp"
    checkpoint = {"model": model.state_dict(), "image_optimizer": image_optimizer.state_dict(),
                  "prior_optimizer": prior_optimizer.state_dict(), "step": 1, "status": "gradient_check_only",
                  "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}}
    torch.save(checkpoint, temporary); os.replace(temporary, directory / f"{checkpoint_name}.pt")
    restored = torch.load(directory / f"{checkpoint_name}.pt", weights_only=False)
    assert restored["step"] == 1 and all(torch.equal(v, restored["model"][k]) for k, v in model.state_dict().items())
    write_json({"status": "prototype_checks_passed_not_trained", "parameters": sum(p.numel() for p in model.parameters()),
                "spline_prior_parameters": sum(p.numel() for p in model.noise_prior.parameters()),
                "noise_conditioning": "per_image" if condition_noise else "shared_across_collection",
                "heldout_target_perturbation_invariance": True, "encoder_decoder_gradient_and_update": True,
                "conditional_KAN_gradient_and_update": True, "checkpoint_roundtrip": True,
                "training_clean_reference_access": False,
                "important_limit": "Residual W2 here is a wiring check. Residuals also contain prediction error; no calibrated noise-learning claim."},
               ROOT / f"results/{checkpoint_name}_checks.json")


def musiq_controls(path, max_seconds):
    import pyiqa
    expected = "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    metric = pyiqa.create_metric("musiq", device="cpu", pretrained_model_path=str(path)).eval().requires_grad_(False)
    output = ROOT / "results/musiq_controls.json"
    rows = json.loads(output.read_text())["rows"] if output.exists() else []
    existing = {(r["pattern"], r["variant"]) for r in rows}
    begin = time.perf_counter()
    for index, (name, clean) in enumerate(patterns().items()):
        rng = np.random.default_rng(701 + index)
        noise = rng.normal(size=clean.shape)
        variants = {"clean_eval_only": clean, "noise_005": clean + .05 * noise,
                    "noise_010": clean + .1 * noise, "flat_mean": np.full_like(clean, clean.mean())}
        for variant, values in variants.items():
            if (name, variant) in existing:
                continue
            if time.perf_counter() - begin > max_seconds:
                return False
            clipped = np.clip(values, 0, 1)
            tensor = torch.tensor(clipped, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
            with torch.no_grad():
                score = float(metric(tensor).item())
            row = {"pattern": name, "variant": variant, "score_higher_is_better": score,
                   "iqa_only_clipped_fraction": float(np.mean(clipped != values)),
                   "clean_evaluation_mse_unclipped": float(np.mean((values - clean) ** 2))}
            rows.append(row)
            write_json({"checkpoint_sha256": expected, "preprocess": "PyIQA MUSIQ KonIQ; grayscale replicated RGB; IQA copy clipped only",
                        "clean_used_for_training": False, "rows": rows}, output)
            print("MUSIQ", name, variant, round(score, 3), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musiq-checkpoint", type=Path)
    parser.add_argument("--max-musiq-seconds", type=float, default=120)
    args = parser.parse_args()
    state = {"stage": "preflight", "status": "running", "full_hybrid_training": "not_started",
             "constraints": {"clean_training_references": False, "observations_per_scene": 1,
                             "noise": "iid Gaussian", "noise_strength": "one shared unknown level"}}
    write_json(state, ROOT / "experiment_state.json")
    if not (ROOT / "results/mechanical_controls.json").exists():
        mathematical_controls()
    if not (ROOT / "results/prototype_shared_checks.json").exists():
        prototype_checks(condition_noise=False)
    musiq_done = musiq_controls(args.musiq_checkpoint, args.max_musiq_seconds) if args.musiq_checkpoint else False
    state.update(status="preflight_complete" if musiq_done else "musiq_pending",
                 next_action="Review metric gate and finalize training protocol; full learned hybrid remains untrained",
                 checkpoint="checkpoints/prototype_shared.pt", musiq_controls_complete=musiq_done,
                 musiq_training_loss="excluded_by_toy_ranking_gate",
                 next_training_stage="masked denoiser baseline, without clean targets or MUSIQ loss")
    write_json(state, ROOT / "experiment_state.json")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
