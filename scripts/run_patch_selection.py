from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
sys.path.insert(0, str(ROOT / "src"))

from denoise_research import torch_optional
from denoise_research.filter_bank import apply_filter_bank
from denoise_research.io import load_image, save_image
from denoise_research.patch_selection import (
    CleanDiscriminatorScorer,
    noise_specific_filter_bank,
    select_patches,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the pretrained clean-discriminator's preferred denoiser patch by patch."
    )
    parser.add_argument("input", type=Path, help="Noisy RGB image")
    parser.add_argument("--output", type=Path, required=True, help="Reconstructed image path")
    parser.add_argument(
        "--noise-type", choices=("awgn", "salt_pepper"), default="awgn",
        help="Known noise model; this selects its matching classical filter bank.",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints" / "netD_Clean_epoch_100.pth",
        help="Archived pretrained clean-discriminator checkpoint.",
    )
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--device", help="PyTorch device; defaults to CUDA when available.")
    parser.add_argument(
        "--selection-map", type=Path,
        help="Optional grayscale map showing which candidate won each image region.",
    )
    args = parser.parse_args()

    torch_optional.require_torch()
    if not args.checkpoint.is_file():
        parser.error(f"Pretrained checkpoint not found: {args.checkpoint}")
    torch = torch_optional.torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, str(device))
    scorer = CleanDiscriminatorScorer(discriminator, str(device))

    noisy = load_image(args.input)
    bank = noise_specific_filter_bank(args.noise_type)
    candidates = apply_filter_bank(noisy, bank)
    result = select_patches(
        candidates,
        scorer,
        scorer.metric_specs,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    save_image(result.image, args.output)

    if args.selection_map:
        # Normalize category IDs for convenient inspection; the JSON sidecar stores
        # the exact category order needed to interpret these values.
        values = result.selection_map.astype(np.float32) / max(1, len(result.filter_names) - 1)
        save_image(values, args.selection_map)

    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "noise_type": args.noise_type,
        "quality_model": "archived CycleGAN clean-domain multiscale PatchGAN discriminator",
        "checkpoint": str(args.checkpoint.resolve()),
        "metric": "mean clean-domain PatchGAN logit; higher is preferred",
        "patch_size": args.patch_size,
        "stride": args.stride,
        "overlap_blending": "Hann weighted overlap-add",
        "filters": list(result.filter_names),
        "selected_patch_counts": result.selected_patch_counts,
    }
    if args.selection_map:
        summary["selection_map"] = str(args.selection_map.resolve())
    summary_path = args.output.with_suffix(args.output.suffix + ".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
