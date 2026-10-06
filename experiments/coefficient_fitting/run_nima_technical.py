from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from denoise_research import torch_optional
from denoise_research.experiment_utils import add_awgn, random_crop
from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics
from run_archive_discriminator import make_archive_fusion


def optimize(torch: object, noisy: object, scorer: object, steps: int) -> tuple[object, dict[str, object]]:
    model = make_archive_fusion(torch, noisy.shape[1]).to(noisy.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history: list[float] = []
    score_history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(noisy)
        quality_score = scorer(reconstruction).mean()
        fidelity = torch.nn.functional.l1_loss(reconstruction, noisy)
        mean_constraint = torch.abs(reconstruction.mean() - noisy.mean())
        # NIMA scores lie in [1, 10]. Normalize only its scale; preserve the
        # archive experiment's fidelity and brightness constraints.
        loss = -quality_score / 10.0 + 10.0 * fidelity + 5.0 * mean_constraint
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
        score_history.append(float(quality_score.detach().cpu()))

    with torch.no_grad():
        raw = model(noisy)
        result = raw.clamp(0.0, 1.0)
        final_score = float(scorer(result).mean().cpu())
        coefficients = model.coefficients().detach().cpu().tolist()
    return result, {
        "coefficients_low_mid_high": coefficients,
        "loss_first": history[0] if history else None,
        "loss_last": history[-1] if history else None,
        "nima_score_first": score_history[0] if score_history else None,
        "nima_score_last_preclip": score_history[-1] if score_history else None,
        "nima_score_final_clipped": final_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit archive low/mid/high coefficients using public NIMA technical weights.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/nima_technical")
    parser.add_argument("--sigma", type=float, default=25.0, help="AWGN standard deviation in 8-bit units.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=4, help="Images per holdout dataset.")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch_optional.require_torch()
    torch = torch_optional.torch
    try:
        from deepinv.loss.metric import NIMA
    except ImportError as exc:
        parser.error(f"Install the quality extra first: pip install -e '.[torch,quality]' ({exc})")
    device = torch.device(args.device)
    # Keep the public checkpoint inside the project rather than the user profile.
    torch.hub.set_dir(str(ROOT / ".cache" / "torch" / "hub"))
    scorer = NIMA(variant="technical", device=device).eval()
    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / "nima_mobilenet_technical_weights.pt"
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest() if weights_path.exists() else None

    discriminator_csv = ROOT / "experiments/coefficient_fitting/results/discriminator_archive_faithful/per_image_results.csv"
    baseline = {}
    if discriminator_csv.exists():
        with discriminator_csv.open(newline="", encoding="utf-8") as handle:
            baseline = {
                row["image"]: row
                for row in csv.DictReader(handle)
                if row["method"] == "archive_with_mean"
            }

    rows: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    images_dir = args.output_dir / "restored"
    for dataset_index, dataset in enumerate(("CBSD68", "Kodak24")):
        paths = image_files(ROOT / "data" / dataset)[: args.max_images]
        if not paths:
            raise RuntimeError(f"No supported images found under {ROOT / 'data' / dataset}")
        for image_index, path in enumerate(paths):
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, dataset_index, image_index]))
            clean = random_crop(load_image(path), args.crop_size, rng)
            noisy = np.clip(add_awgn(clean, args.sigma, rng), 0.0, 1.0)
            tensor = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            image_id = f"{dataset}/{path.name}"

            rows.append({"image": image_id, "method": "noisy", **full_reference_metrics(clean, noisy)})
            estimate, record = optimize(torch, tensor, scorer, args.steps)
            estimate_array = estimate.squeeze(0).permute(1, 2, 0).cpu().numpy()
            rows.append({"image": image_id, "method": "nima_technical", **full_reference_metrics(clean, estimate_array)})
            if image_id in baseline:
                rows.append({
                    "image": image_id,
                    "method": "cyclegan_discriminator_with_mean",
                    **{key: float(baseline[image_id][key]) for key in ("mse", "mae", "psnr", "ssim")},
                })
            image_out = images_dir / dataset / f"{path.stem}_nima.png"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.uint8(np.clip(estimate_array, 0, 1) * 255 + 0.5)).save(image_out)
            fits.append({"image": image_id, **record})
            print(f"{image_id}: coeffs={record['coefficients_low_mid_high']}, NIMA={record['nima_score_final_clipped']:.4f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "fitted_coefficients.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "quality_model": "NIMA technical (MobileNet, TID2013-trained), public converted author weights",
            "quality_model_type": "no-reference image-quality assessor; not a discriminator",
            "weights_file": str(weights_path),
            "weights_sha256": weights_hash,
            "bands": "archive Gaussian low + averaged Gabor mid + residual high",
            "coefficient_rule": "absolute values normalized to sum to 1",
            "noise": {"type": "AWGN then clip to [0,1]", "sigma_8bit": args.sigma, "seed": args.seed},
            "crop_size": args.crop_size,
            "steps": args.steps,
            "learning_rate": 0.01,
            "objective": "-NIMA_score/10 + 10*L1(reconstruction,noisy) + 5*abs(mean(reconstruction)-mean(noisy))",
            "device": str(device),
            "fits": fits,
        }, handle, indent=2)
    print(f"Wrote NIMA technical results to {args.output_dir}")


if __name__ == "__main__":
    main()
