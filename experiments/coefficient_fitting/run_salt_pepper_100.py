from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(EXPERIMENTS))

from denoise_research import torch_optional
from denoise_research.io import image_files, load_image, save_image
from denoise_research.metrics import full_reference_metrics
from denoise_research.operators import median_filter
from denoise_research.pipelines import complementary_bands
from run_discriminator_100_compare_visual import optimize_batch as optimize_discriminator_batch
from run_nima_100_visual import optimize_batch as optimize_nima_batch


def selected_crops(seed: int, crop_size: int, amount: float) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for dataset_index, (dataset, count) in enumerate((("CBSD68", 50), ("Kodak24", 50))):
        paths = image_files(ROOT / "data" / dataset)
        if not paths:
            raise RuntimeError(f"No images found under {ROOT / 'data' / dataset}")
        for local_index in range(count):
            source_index = local_index % len(paths)
            repeat_index = local_index // len(paths)
            path = paths[source_index]
            clean_full = load_image(path)
            height, width = clean_full.shape[:2]
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, dataset_index, source_index, repeat_index])
            )
            top = int(rng.integers(0, height - crop_size + 1))
            left = int(rng.integers(0, width - crop_size + 1))
            clean = clean_full[top : top + crop_size, left : left + crop_size].copy()
            selector = rng.random((crop_size, crop_size))
            noisy = clean.copy()
            noisy[selector < amount / 2.0] = 0.0
            noisy[(selector >= amount / 2.0) & (selector < amount)] = 1.0
            records.append({
                "dataset": dataset,
                "path": path,
                "repeat": repeat_index,
                "top": top,
                "left": left,
                "clean": clean.astype(np.float32),
                "noisy": noisy.astype(np.float32),
                "corrupted_fraction": float(np.mean(selector < amount)),
            })
    return records


def sure_and_oracle(noisy: np.ndarray, clean: np.ndarray, sigma_proxy: float) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    bands = np.stack(complementary_bands(noisy, 1.0, 2.0), axis=0)
    candidates = np.asarray(list(itertools.product(np.linspace(0.0, 2.0, 5), repeat=3)), dtype=np.float32)
    candidate_images = np.tensordot(candidates, bands, axes=(1, 0)).astype(np.float32)

    rng = np.random.default_rng(271828)
    probe = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=noisy.shape)
    probe_bands = np.stack(complementary_bands(probe, 1.0, 2.0), axis=0)
    divergence = np.mean(probe[None, ...] * probe_bands, axis=(1, 2, 3))
    residual_mse = np.mean((candidate_images - noisy[None, ...]) ** 2, axis=(1, 2, 3))
    sure_scores = residual_mse + 2.0 * sigma_proxy**2 * (candidates @ divergence) - sigma_proxy**2
    sure_index = int(np.argmin(sure_scores))

    oracle_scores = np.mean((candidate_images - clean[None, ...]) ** 2, axis=(1, 2, 3))
    oracle_index = int(np.argmin(oracle_scores))
    return (
        np.clip(candidate_images[sure_index], 0.0, 1.0),
        np.clip(candidate_images[oracle_index], 0.0, 1.0),
        candidates[sure_index].tolist(),
        candidates[oracle_index].tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare denoising methods on 100 salt-and-pepper-corrupted crops.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/netD_Clean_epoch_100.pth")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENTS / "results/salt_pepper_100")
    parser.add_argument("--amount", type=float, default=0.10, help="Fraction of whole RGB pixels replaced by salt or pepper.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0.0 < args.amount < 1.0:
        parser.error("--amount must lie between 0 and 1")

    torch_optional.require_torch()
    torch = torch_optional.torch
    from deepinv.loss.metric import NIMA

    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    nima = NIMA(variant="technical", device=args.device).eval()
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, args.device)
    examples = selected_crops(args.seed, args.crop_size, args.amount)
    sigma_proxy = float(np.sqrt(args.amount) / 2.0)
    image_dir = args.output_dir / "images"
    rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []

    for start in range(0, len(examples), args.batch_size):
        group = examples[start : start + args.batch_size]
        noisy_batch = np.stack([item["noisy"] for item in group])
        tensor = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(args.device)
        nima_tensor, nima_fits = optimize_nima_batch(torch, tensor, nima, args.steps)
        disc_tensor, disc_coefficients = optimize_discriminator_batch(
            torch, tensor, discriminator, args.steps
        )
        nima_batch = nima_tensor.permute(0, 2, 3, 1).cpu().numpy()
        disc_batch = disc_tensor.permute(0, 2, 3, 1).cpu().numpy()

        for offset, (item, nima_out, disc_out, nima_fit, disc_weights) in enumerate(
            zip(group, nima_batch, disc_batch, nima_fits, disc_coefficients)
        ):
            index = start + offset + 1
            clean = item["clean"]
            noisy = item["noisy"]
            median3 = np.clip(median_filter(noisy, 3), 0.0, 1.0)
            median5 = np.clip(median_filter(noisy, 5), 0.0, 1.0)
            sure, oracle, sure_weights, oracle_weights = sure_and_oracle(
                noisy, clean, sigma_proxy
            )
            outputs = {
                "noisy": noisy,
                "median_3x3": median3,
                "median_5x5": median5,
                "nima_technical": nima_out,
                "archive_discriminator": disc_out,
                "sure_gaussian_proxy_mismatched": sure,
                "oracle_three_band_grid": oracle,
            }
            source = item["path"]
            key = f"{index:03d}_{item['dataset']}_{source.stem}_r{item['repeat']}"
            save_image(clean, image_dir / f"{key}_clean.png")
            for method, output in outputs.items():
                save_image(output, image_dir / f"{key}_{method}.png")
                rows.append({
                    "example": index,
                    "dataset": item["dataset"],
                    "image": source.name,
                    "top": item["top"],
                    "left": item["left"],
                    "corrupted_fraction": item["corrupted_fraction"],
                    "method": method,
                    **full_reference_metrics(clean, output),
                })
            fit_rows.append({
                "example": index,
                "dataset": item["dataset"],
                "image": source.name,
                "nima_low_mid_high": nima_fit["coefficients_low_mid_high"],
                "discriminator_low_mid_high": disc_weights,
                "sure_low_mid_high": sure_weights,
                "oracle_low_mid_high": oracle_weights,
            })
        print(f"Completed {min(start + args.batch_size, len(examples))}/100 crops", flush=True)

    methods = [
        "noisy", "median_3x3", "median_5x5", "nima_technical",
        "archive_discriminator", "sure_gaussian_proxy_mismatched", "oracle_three_band_grid",
    ]
    summary: list[dict[str, object]] = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary.append({
            "method": method,
            "images": len(selected),
            "mean_psnr": float(np.mean([row["psnr"] for row in selected])),
            "mean_ssim": float(np.mean([row["ssim"] for row in selected])),
            "mean_mse": float(np.mean([row["mse"] for row in selected])),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "count": 100,
        "datasets": {"CBSD68": 50, "Kodak24": 50},
        "noise": {
            "type": "whole-pixel salt-and-pepper",
            "amount": args.amount,
            "salt_fraction": args.amount / 2.0,
            "pepper_fraction": args.amount / 2.0,
            "seed": args.seed,
        },
        "crop_size": args.crop_size,
        "steps": args.steps,
        "sure_warning": "Classical Gaussian SURE is invalid for salt-and-pepper noise. This stress test uses sigma_proxy=sqrt(amount)/2 and is not a deployable SURE result.",
        "sure_sigma_proxy": sigma_proxy,
        "oracle_warning": "The oracle three-band grid uses clean references and is a representational ceiling only.",
        "fits": fit_rows,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
