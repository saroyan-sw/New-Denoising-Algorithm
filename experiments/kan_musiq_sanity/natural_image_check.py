"""Fixed natural-photo MUSIQ calibration and RGB prototype checks; no full training."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import data
import skimage

from models import MaskedImageKAN, checker_mask, blind_loss
from preflight import write_json

NAMES = ["astronaut", "coffee", "chelsea", "camera"]
SEEDS = [2026111601, 2026111602, 2026111603]
WEIGHT_SHA = "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa"


def photo(name):
    image = getattr(data, name)().astype(np.float32) / 255.
    return np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image


def cases(reference, seed):
    rng = np.random.default_rng(seed)
    unit = rng.normal(size=reference.shape).astype(np.float32)
    noisy = reference + .05 * unit
    return {
        "reference": reference,
        "noise_025": reference + .025 * unit,
        "noise_050": noisy,
        "noise_100": reference + .1 * unit,
        "smoothed_noise_050": gaussian_filter(noisy, sigma=(1., 1., 0)),
        "independent_subtraction_050": noisy - .05 * rng.normal(size=reference.shape).astype(np.float32),
        "blurred_reference": gaussian_filter(reference, sigma=(1., 1., 0)),
        "flat_mean": np.broadcast_to(reference.mean(axis=(0, 1), keepdims=True), reference.shape).copy()
    }


def rgb_checks():
    torch.manual_seed(491)
    model = MaskedImageKAN(channels=3)
    noisy = torch.randn(2, 3, 32, 32) * .05 + .5
    mask = checker_mask(noisy)
    prediction, code = model(noisy, mask)
    changed, changed_code = model(noisy + 100 * mask, mask)
    assert prediction.shape == noisy.shape
    assert torch.equal(prediction, changed) and torch.equal(code, changed_code)
    image_optimizer = torch.optim.Adam([*model.encoder.parameters(), *model.context.parameters(), *model.decoder.parameters()])
    image_optimizer.zero_grad(); blind_loss(prediction, noisy, mask).backward(); image_optimizer.step()
    with torch.no_grad():
        prediction, code = model(noisy, mask)
        residuals = torch.stack([(noisy - prediction)[b][mask[b].expand_as(noisy[b]).bool()][:256] for b in range(2)])
    noise_optimizer = torch.optim.Adam(model.noise_prior.parameters())
    samples = model.sample_noise(torch.rand(2, 256, 1), code.detach())
    loss = (samples.sort(1).values - residuals.sort(1).values).square().mean()
    noise_optimizer.zero_grad(); loss.backward(); noise_optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    cp = ROOT / "checkpoints/prototype_rgb.pt"
    cp.parent.mkdir(exist_ok=True)
    temporary = cp.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "channels": 3, "context_dim": 4,
                "image_optimizer": image_optimizer.state_dict(), "noise_optimizer": noise_optimizer.state_dict(),
                "status": "one_update_wiring_check_only", "step": 1, "torch_rng": torch.get_rng_state()}, temporary)
    os.replace(temporary, cp)
    loaded = torch.load(cp, weights_only=False)
    assert all(torch.equal(value, loaded["model"][key]) for key, value in model.state_dict().items())
    write_json({"status": "passed", "channels": 3, "parameters": sum(p.numel() for p in model.parameters()),
                "shared_KAN_parameters": sum(p.numel() for p in model.noise_prior.parameters()),
                "target_exclusion": True, "finite_gradients_and_updates": True, "checkpoint_roundtrip": True,
                "full_training": False}, ROOT / "results/rgb_prototype_checks.json")


def quality_gradient_check(metric):
    metric.as_loss = True
    sample = np.clip(cases(photo("astronaut"), SEEDS[0])["noise_050"], 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(sample.transpose(2, 0, 1)))[None].requires_grad_(True)
    score = metric(tensor).mean()
    gradient = torch.autograd.grad(score, tensor)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(not parameter.requires_grad for parameter in metric.parameters())
    write_json({"status": "passed", "metric_frozen": True, "score": float(score.item()),
                "input_gradient_norm": float(gradient.norm().item()),
                "input_gradient_finite": True, "optimizer_steps": 0,
                "interpretation": "Differentiability check only; no metric optimization or denoising training."},
               ROOT / "results/natural_quality_gradient.json")


def summarize(rows):
    means = {}
    for name in NAMES:
        means[name] = {}
        for variant in cases(photo(name), SEEDS[0]):
            values = [r["musiq"] for r in rows if r["image"] == name and r["variant"] == variant]
            means[name][variant] = float(np.mean(values))
    primary_gate = all(
        m["reference"] > m["noise_050"] > m["noise_100"] and m["reference"] > m["flat_mean"]
        for m in means.values()
    )
    monotonic = {name: m["reference"] > m["noise_025"] > m["noise_050"] > m["noise_100"]
                 for name, m in means.items()}
    summary = {"status": "complete", "scores_mean_over_noise_seeds": means,
               "primary_auxiliary_loss_gate": primary_gate,
               "strict_monotonicity_all_levels_by_image": monotonic,
               "interpretation": "Domain calibration only; not proof of a denoising model or safe standalone metric optimization.",
               "all_cases_reported": True, "full_hybrid_training": "not_started"}
    write_json(summary, ROOT / "results/natural_summary.json")
    fig, axes = plt.subplots(4, 4, figsize=(13, 13), constrained_layout=True)
    variants = ["reference", "noise_050", "smoothed_noise_050", "independent_subtraction_050"]
    for r, name in enumerate(NAMES):
        variants_images = cases(photo(name), SEEDS[0])
        for c, variant in enumerate(variants):
            row = next(v for v in rows if v["image"] == name and v["variant"] == variant and v["seed"] in (0, SEEDS[0]))
            axes[r, c].imshow(np.clip(variants_images[variant], 0, 1))
            axes[r, c].set_title(f"{name}: {variant}\nMUSIQ {row['musiq']:.2f} | MSE {row['mse_to_reference_raw']:.4f}", fontsize=9)
            axes[r, c].axis("off")
    fig.savefig(ROOT / "results/natural_image_comparison.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name, values in means.items():
        ax.plot([0, .025, .05, .1], [values[v] for v in ["reference", "noise_025", "noise_050", "noise_100"]], "o-", label=name)
    ax.set(xlabel="Added Gaussian noise standard deviation", ylabel="MUSIQ (higher is better)", title="Fixed natural-photo calibration: three noise seeds")
    ax.legend(); fig.savefig(ROOT / "results/natural_musiq_curve.png", dpi=150); plt.close(fig)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musiq-checkpoint", type=Path, required=True)
    parser.add_argument("--max-runtime-seconds", type=float, default=300)
    args = parser.parse_args()
    protocol = {"images_fixed_before_scoring": NAMES, "source": f"skimage.data {skimage.__version__}",
                "seeds": SEEDS, "noise_levels": [.025, .05, .1], "raw_arrays_clipped": False,
                "iqa_copy_clipped": True, "external_resize": False, "reference_use": "evaluation_only",
                "primary_gate_fixed_before_scoring": "For every photo: mean reference > noise050 > noise100, and reference > flat mean",
                "role": "metric-development calibration, not unseen model holdout",
                "inputs": {n: {"shape": list(photo(n).shape), "sha256_float32": hashlib.sha256(photo(n).tobytes()).hexdigest()} for n in NAMES}}
    write_json(protocol, ROOT / "results/natural_protocol.json")
    if not (ROOT / "results/rgb_prototype_checks.json").exists():
        rgb_checks()
    result_path = ROOT / "results/natural_scores.json"
    rows = json.loads(result_path.read_text())["rows"] if result_path.exists() else []
    existing = {(r["image"], r["variant"], r["seed"]) for r in rows}
    assert hashlib.sha256(args.musiq_checkpoint.read_bytes()).hexdigest() == WEIGHT_SHA
    import pyiqa
    metric = pyiqa.create_metric("musiq", device="cpu", pretrained_model_path=str(args.musiq_checkpoint)).eval().requires_grad_(False)
    begin = time.perf_counter()
    for name in NAMES:
        reference = photo(name)
        for seed in SEEDS:
            for variant, value in cases(reference, seed).items():
                record_seed = 0 if variant in {"reference", "blurred_reference", "flat_mean"} else seed
                key = (name, variant, record_seed)
                if key in existing:
                    continue
                if time.perf_counter() - begin > args.max_runtime_seconds:
                    print("Session budget reached; rerun this command to continue saved scores.")
                    return
                clipped = np.clip(value, 0, 1)
                tensor = torch.from_numpy(np.ascontiguousarray(clipped.transpose(2, 0, 1)))[None]
                with torch.no_grad():
                    score = float(metric(tensor).item())
                row = {"image": name, "variant": variant, "seed": record_seed, "musiq": score,
                       "mse_to_reference_raw": float(np.mean((value.astype(np.float64) - reference) ** 2)),
                       "iqa_clipped_fraction": float(np.mean(clipped != value))}
                rows.append(row); existing.add(key)
                write_json({"checkpoint_sha256": WEIGHT_SHA, "rows": rows}, result_path)
                print(name, variant, record_seed, f"{score:.3f}", flush=True)
    assert len(rows) == 72, len(rows)
    if not (ROOT / "results/natural_quality_gradient.json").exists():
        quality_gradient_check(metric)
    summarize(rows)


if __name__ == "__main__":
    main()
