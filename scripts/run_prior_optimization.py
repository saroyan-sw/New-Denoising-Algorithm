from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.io import load_image, save_image
from denoise_research.pipelines import overlap_add_apply
from denoise_research import torch_optional


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the corrected three-band optimizer with a frozen clean discriminator"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/complementary_bands.json"))
    parser.add_argument("--mode", choices=("global", "patch"), default="global")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    torch_optional.require_torch()
    torch = torch_optional.torch
    config = json.loads(args.config.read_text(encoding="utf-8"))
    optimization = config.get("optimization", {})
    use_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    discriminator = torch_optional.load_clean_discriminator(args.checkpoint, str(use_device))
    records: list[dict[str, object]] = []

    def process(array: np.ndarray) -> np.ndarray:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        tensor = torch.from_numpy(contiguous).permute(2, 0, 1).unsqueeze(0).to(use_device)
        restored, record = torch_optional.optimize_with_frozen_prior(
            tensor,
            discriminator,
            steps=int(optimization.get("steps", 150)),
            learning_rate=float(optimization.get("learning_rate", 0.01)),
            fidelity_weight=float(optimization.get("fidelity_weight", 10.0)),
            mean_weight=float(optimization.get("mean_weight", 5.0)),
            fine_sigma=float(config.get("fine_sigma", 1.0)),
            coarse_sigma=float(config.get("coarse_sigma", 2.0)),
        )
        records.append(record)
        return restored.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    image = load_image(args.input)
    if args.mode == "patch":
        if args.patch_size < 64:
            parser.error("PatchGAN optimization requires --patch-size of at least 64")
        result = overlap_add_apply(image, process, args.patch_size, args.stride)
    else:
        result = process(image)
    save_image(result, args.output)

    metadata = args.metadata or args.output.with_suffix(".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "mode": args.mode,
        "patch_size": args.patch_size if args.mode == "patch" else None,
        "stride": args.stride if args.mode == "patch" else None,
        "config": config,
        "optimizations": records,
    }
    metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.output} and {metadata}")


if __name__ == "__main__":
    main()

