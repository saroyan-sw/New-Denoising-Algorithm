from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from denoise_research import torch_optional
from denoise_research.experiment_utils import add_awgn, random_crop
from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics


def make_archive_fusion(torch: object, channels: int = 3) -> object:
    """Recreate archive_code_only/unsupervised_optimization.py SelfFuseNet bands."""
    nn = torch.nn
    functional = torch.nn.functional

    class ArchiveFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            size, sigma = 9, 2.0
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            profile = torch.exp(-(coords**2) / (2 * sigma**2))
            profile /= profile.sum()
            gaussian = (profile[:, None] * profile[None, :]).view(1, 1, size, size)
            gaussian = gaussian.repeat(channels, 1, 1, 1)
            self.register_buffer("blur_kernel", gaussian)

            theta_values = [(theta / 4) * np.pi for theta in range(8)]
            gabor_kernels = [
                cv2.getGaborKernel(
                    (5, 5), 3.0, theta, np.pi / 4, 0.5, 0, ktype=cv2.CV_32F
                )
                for theta in theta_values
            ]
            mean_gabor = np.mean(gabor_kernels, axis=0)
            gabor = torch.from_numpy(mean_gabor).view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
            self.register_buffer("gabor_kernel", gabor)
            self.c_low = nn.Parameter(torch.tensor(0.33))
            self.c_mid = nn.Parameter(torch.tensor(0.33))
            self.c_high = nn.Parameter(torch.tensor(0.33))

        def coefficients(self) -> object:
            values = torch.stack((self.c_low, self.c_mid, self.c_high)).abs()
            return values / (values.sum() + 1e-8)

        def bands(self, image: object) -> tuple[object, object, object]:
            low = functional.conv2d(image, self.blur_kernel, padding=4, groups=channels)
            mid = functional.conv2d(image, self.gabor_kernel, padding=2, groups=channels)
            high = image - low - mid
            return low, mid, high

        def forward(self, image: object) -> object:
            low, mid, high = self.bands(image)
            coeffs = self.coefficients()
            return coeffs[0] * low + coeffs[1] * mid + coeffs[2] * high

    return ArchiveFusion()


def optimize(
    torch: object,
    noisy_tensor: object,
    discriminator: object,
    *,
    steps: int,
    use_mean_constraint: bool,
) -> tuple[object, dict[str, object]]:
    functional = torch.nn.functional
    model = make_archive_fusion(torch, noisy_tensor.shape[1]).to(noisy_tensor.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_history: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        reconstruction = model(noisy_tensor)
        predictions, _ = discriminator(2.0 * reconstruction - 1.0)
        adversarial = sum(
            functional.mse_loss(prediction, torch.ones_like(prediction))
            for prediction in predictions
        )
        fidelity = functional.l1_loss(reconstruction, noisy_tensor)
        loss = adversarial + 10.0 * fidelity
        if use_mean_constraint:
            loss = loss + 5.0 * torch.abs(reconstruction.mean() - noisy_tensor.mean())
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        result = model(noisy_tensor).clamp(0.0, 1.0)
        coefficients = model.coefficients().detach().cpu().tolist()
    return result, {"coefficients_low_mid_high": coefficients, "loss": loss_history}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun checkpoint-guided coefficient fitting with the archived Gabor-band model."
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/netD_Clean_epoch_100.pth")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "results/discriminator_archive_faithful",
    )
    parser.add_argument("--sigma", type=float, default=25.0, help="AWGN standard deviation in 8-bit units.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=4, help="Images per holdout dataset.")
    parser.add_argument("--steps", type=int, default=150)
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
    fits: list[dict[str, object]] = []

    for dataset_index, dataset in enumerate(("CBSD68", "Kodak24")):
        data_root = ROOT / "data" / dataset
        paths = image_files(data_root)[: args.max_images]
        if not paths:
            raise RuntimeError(f"No supported images found under {data_root}")
        for image_index, path in enumerate(paths):
            rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, dataset_index, image_index])
            )
            clean = random_crop(load_image(path), args.crop_size, rng)
            # Match the archive's clipped noisy-input range while retaining sigma=25/255.
            noisy = np.clip(add_awgn(clean, args.sigma, rng), 0.0, 1.0)
            noisy_tensor = torch.from_numpy(np.ascontiguousarray(noisy, dtype=np.float32))
            noisy_tensor = noisy_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            image_id = f"{dataset}/{path.name}"

            rows.append({"image": image_id, "method": "noisy", **full_reference_metrics(clean, noisy)})
            for label, mean_constraint in (("archive_no_mean", False), ("archive_with_mean", True)):
                estimate, record = optimize(
                    torch,
                    noisy_tensor,
                    discriminator,
                    steps=args.steps,
                    use_mean_constraint=mean_constraint,
                )
                estimate_array = estimate.squeeze(0).permute(1, 2, 0).cpu().numpy()
                rows.append(
                    {"image": image_id, "method": label, **full_reference_metrics(clean, estimate_array)}
                )
                history = record.pop("loss")
                fits.append(
                    {
                        "image": image_id,
                        "method": label,
                        **record,
                        "loss_first": history[0] if history else None,
                        "loss_last": history[-1] if history else None,
                    }
                )
                print(
                    f"{image_id} {label}: coefficients="
                    f"{record['coefficients_low_mid_high']}",
                    flush=True,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "fitted_coefficients.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "model": "archived Gaussian low + averaged Gabor mid + residual high",
                "coefficient_rule": "absolute values normalized to sum to 1",
                "noise": {
                    "type": "AWGN then clip to [0,1]",
                    "sigma_8bit": args.sigma,
                    "seed": args.seed,
                },
                "crop_size": args.crop_size,
                "steps": args.steps,
                "learning_rate": 0.01,
                "fidelity_weight": 10.0,
                "mean_constraint_weight": 5.0,
                "device": str(device),
                "fits": fits,
            },
            handle,
            indent=2,
        )
    print(f"Wrote archive-faithful results to {args.output_dir}")


if __name__ == "__main__":
    main()
