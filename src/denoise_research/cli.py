from __future__ import annotations

import argparse
from pathlib import Path

from .config import PaperInspiredConfig
from .evaluation import benchmark_images, write_csv
from .io import image_files, load_image, save_image
from .metrics import mse
from .pipelines import PaperInspiredPipeline
from .visualization import comparison_grid


def baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the audited paper-inspired deterministic baseline")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path, help="Optional clean reference for oracle grid-search diagnostics")
    parser.add_argument("--grid", type=Path, help="Optional labeled comparison image")
    args = parser.parse_args(argv)

    config = PaperInspiredConfig.from_json(args.config) if args.config else PaperInspiredConfig()
    pipeline = PaperInspiredPipeline(config)
    image = load_image(args.input)
    if args.reference:
        reference = load_image(args.reference)
        result, weights, score = pipeline.grid_search(image, lambda candidate: mse(reference, candidate))
        print(f"oracle_weights={weights} oracle_mse={score:.8f}")
    else:
        reference = None
        result = pipeline.run(image)
    save_image(result, args.output)
    if args.grid:
        panels = {"input": image, "restored": result}
        if reference is not None:
            panels = {"reference": reference, **panels}
        comparison_grid(panels, args.grid)


def benchmark_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark the audited baseline on reproducible degradations")
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = image_files(args.image_directory)
    if not paths:
        parser.error(f"No supported images found under {args.image_directory}")
    rows = benchmark_images(paths, seed=args.seed)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")

