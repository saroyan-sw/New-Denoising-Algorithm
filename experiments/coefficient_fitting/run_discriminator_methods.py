from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from denoise_research import torch_optional
from denoise_research.experiment_utils import add_awgn, random_crop
from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pilot per-image three-band coefficient fitting with a frozen clean discriminator."
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/netD_Clean_epoch_100.pth")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/discriminator")
    parser.add_argument("--sigma", type=float, default=25.0, help="AWGN standard deviation in 8-bit units.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=4, help="Images per evaluation dataset.")
    parser.add_argument("--steps", type=int, default=8, help="Optimization steps per image (short CPU pilot).")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch_optional.require_torch()
    torch = torch_optional.torch
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    device = torch.device(args.device)
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, str(device))
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    for dataset in ("CBSD68", "Kodak24"):
        root = ROOT / "data" / dataset
        paths = image_files(root)[: args.max_images]
        if not paths:
            raise RuntimeError(f"No supported images found under {root}")
        for index, path in enumerate(paths):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0 if dataset == "CBSD68" else 1, index]))
            clean = random_crop(load_image(path), args.crop_size, rng)
            noisy = add_awgn(clean, args.sigma, rng)
            clean_tensor = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))
            clean_tensor = clean_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            restored, fit = torch_optional.optimize_with_frozen_prior(
                clean_tensor,
                discriminator,
                steps=args.steps,
            )
            estimate = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            identity = f"{dataset}/{path.name}"
            for method, image in (("noisy", noisy), ("discriminator_fitted", estimate)):
                rows.append({"image": identity, "method": method, **full_reference_metrics(clean, image)})
            records.append(
                {
                    "image": identity,
                    "weights_low_mid_high": fit["weights"],
                    "loss_first": fit["loss"][0] if fit["loss"] else None,
                    "loss_last": fit["loss"][-1] if fit["loss"] else None,
                }
            )
            print(f"{identity}: fitted weights={fit['weights']}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "fitted_weights.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "noise": {"type": "unclipped_awgn", "sigma_8bit": args.sigma, "seed": args.seed},
                "crop_size": args.crop_size,
                "steps": args.steps,
                "device": str(device),
                "records": records,
                "results_are_pilot": True,
            },
            handle,
            indent=2,
        )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
