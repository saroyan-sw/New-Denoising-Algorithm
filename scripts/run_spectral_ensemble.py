from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterator

import numpy as np

# Keep PyIQA's downloaded model weights inside this project by default.
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "torch"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.experiment_utils import add_awgn, random_crop, write_csv, write_json
from denoise_research.filter_bank import apply_filter_bank, default_filter_bank, select_filter_bank
from denoise_research.io import image_files, load_image, save_image
from denoise_research.metrics import full_reference_metrics
from denoise_research.nr_metrics import PyIQACollection, parse_metric_spec
from denoise_research.spectral_ensemble import (
    combine_affinities,
    fit_spectral_weights,
    fuse_candidates,
    metric_to_affinity,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a HITS-like spectral ensemble of deterministic denoisers."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="Fit global weights on clean calibration images.")
    fit.add_argument("clean_directory", type=Path)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--sigma", type=float, default=25.0, help="AWGN sigma in 8-bit units.")
    fit.add_argument("--seed", type=int, default=20260917)
    fit.add_argument("--crop-size", type=int, default=256)
    fit.add_argument("--max-images", type=int)
    fit.add_argument(
        "--metric",
        action="append",
        default=None,
        metavar="NAME:DIRECTION",
        help="PyIQA NR metric, e.g. brisque:lower. May be repeated.",
    )
    fit.add_argument("--device", help="PyTorch device; default chooses CUDA when available.")
    fit.add_argument("--affinity", choices=("rank", "softmax"), default="rank")
    fit.add_argument("--temperature", type=float, default=1.0)
    fit.add_argument("--save-examples", type=int, default=3)

    evaluate = commands.add_parser("evaluate", help="Apply frozen weights to a clean test set.")
    evaluate.add_argument("clean_directory", type=Path)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--sigma", type=float, help="Override the sigma saved during fitting.")
    evaluate.add_argument("--seed", type=int, default=20260917)
    evaluate.add_argument("--crop-size", type=int, default=0, help="Zero evaluates full images.")
    evaluate.add_argument("--max-images", type=int)
    evaluate.add_argument("--save-images", action="store_true")
    return parser


def _paths(directory: Path, maximum: int | None) -> list[Path]:
    paths = image_files(directory)
    if maximum is not None:
        if maximum < 1:
            raise ValueError("--max-images must be positive")
        paths = paths[:maximum]
    if not paths:
        raise ValueError(f"No supported images found under {directory}")
    return paths


def _samples(
    paths: list[Path], root: Path, *, sigma: float, seed: int, crop_size: int
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    for index, path in enumerate(paths):
        # A separate deterministic stream per image makes reruns and the second fit pass identical.
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        clean = random_crop(load_image(path), crop_size, rng)
        noisy = add_awgn(clean, sigma, rng)
        yield path.relative_to(root).as_posix(), clean, noisy


def _metric_row(image_id: str, method: str, clean: np.ndarray, estimate: np.ndarray) -> dict:
    return {"image": image_id, "method": method, **full_reference_metrics(clean, estimate)}


def _summary(rows: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        result[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in ("mse", "mae", "psnr", "ssim")
        }
    return result


def fit(args: argparse.Namespace) -> None:
    paths = _paths(args.clean_directory, args.max_images)
    bank = default_filter_bank()
    names = [spec.name for spec in bank]
    metric_values = args.metric or ["brisque:lower", "niqe:lower"]
    specs = [parse_metric_spec(value) for value in metric_values]
    scorer = PyIQACollection(specs, device=args.device)

    matrices = {
        spec.name: np.empty((len(paths), len(bank)), dtype=np.float64) for spec in specs
    }
    score_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for image_index, (image_id, clean, noisy) in enumerate(
        _samples(
            paths,
            args.clean_directory,
            sigma=args.sigma,
            seed=args.seed,
            crop_size=args.crop_size,
        )
    ):
        candidates = apply_filter_bank(noisy, bank)
        for filter_index, name in enumerate(names):
            values = scorer.score(candidates[name])
            for spec in specs:
                matrices[spec.name][image_index, filter_index] = values[spec.name]
            score_rows.append({"image": image_id, "filter": name, **values})
            candidate_rows.append(_metric_row(image_id, name, clean, candidates[name]))

    affinity_by_metric = {
        spec.name: metric_to_affinity(
            matrices[spec.name],
            lower_is_better=spec.lower_is_better,
            mode=args.affinity,
            temperature=args.temperature,
        )
        for spec in specs
    }
    affinity = combine_affinities(list(affinity_by_metric.values()))
    spectral = fit_spectral_weights(affinity)
    weight_map = {name: float(value) for name, value in zip(names, spectral.filter_weights)}
    uniform_map = {name: 1.0 / len(names) for name in names}

    for row_index, row in enumerate(score_rows):
        image_index, filter_index = divmod(row_index, len(names))
        row["combined_affinity"] = float(affinity[image_index, filter_index])
    fused_rows: list[dict] = []
    for example_index, (image_id, clean, noisy) in enumerate(
        _samples(
            paths,
            args.clean_directory,
            sigma=args.sigma,
            seed=args.seed,
            crop_size=args.crop_size,
        )
    ):
        candidates = apply_filter_bank(noisy, bank)
        fused = fuse_candidates(candidates, weight_map)
        uniform = fuse_candidates(candidates, uniform_map)
        fused_rows.append(_metric_row(image_id, "spectral_fusion", clean, fused))
        fused_rows.append(_metric_row(image_id, "uniform_fusion", clean, uniform))
        if example_index < args.save_examples:
            stem = f"{example_index:03d}_{Path(image_id).stem}"
            save_image(clean, args.output_dir / "examples" / f"{stem}_clean.png")
            save_image(noisy, args.output_dir / "examples" / f"{stem}_noisy.png")
            save_image(fused, args.output_dir / "examples" / f"{stem}_fused.png")

    all_metric_rows = candidate_rows + fused_rows
    payload = {
        "method": "damped_bipartite_spectral_consensus",
        "warning": "These are spectral consensus coefficients, not error-optimal coefficients.",
        "fit_dataset": str(args.clean_directory.resolve()),
        "image_ids": [path.relative_to(args.clean_directory).as_posix() for path in paths],
        "noise": {"type": "awgn", "sigma_255": args.sigma, "clip_noisy_input": False},
        "seed": args.seed,
        "crop_size": args.crop_size,
        "filters": names,
        "filter_weights": weight_map,
        "image_weights": [float(value) for value in spectral.image_weights],
        "metrics": [
            {"name": spec.name, "lower_is_better": spec.lower_is_better} for spec in specs
        ],
        "affinity": {"mode": args.affinity, "temperature": args.temperature},
        "solver": {
            "iterations": spectral.iterations,
            "converged": spectral.converged,
            "damping": 0.95,
        },
        "fit_full_reference_summary": _summary(all_metric_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "weights.json", payload)
    write_csv(
        args.output_dir / "nr_scores.csv",
        score_rows,
        ["image", "filter", *[spec.name for spec in specs], "combined_affinity"],
    )
    write_csv(
        args.output_dir / "fit_metrics.csv",
        all_metric_rows,
        ["image", "method", "mse", "mae", "psnr", "ssim"],
    )
    print(json.dumps({"weights": weight_map, "output": str(args.output_dir)}, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    with args.weights.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    names = list(payload["filters"])
    weight_map = {name: float(payload["filter_weights"][name]) for name in names}
    uniform_map = {name: 1.0 / len(names) for name in names}
    bank = select_filter_bank(names)
    saved_sigma = float(payload["noise"]["sigma_255"])
    sigma = saved_sigma if args.sigma is None else args.sigma
    paths = _paths(args.clean_directory, args.max_images)

    rows: list[dict] = []
    for image_index, (image_id, clean, noisy) in enumerate(
        _samples(
            paths,
            args.clean_directory,
            sigma=sigma,
            seed=args.seed,
            crop_size=args.crop_size,
        )
    ):
        candidates = apply_filter_bank(noisy, bank)
        fused = fuse_candidates(candidates, weight_map)
        uniform = fuse_candidates(candidates, uniform_map)
        for name, candidate in candidates.items():
            rows.append(_metric_row(image_id, name, clean, candidate))
        rows.append(_metric_row(image_id, "spectral_fusion", clean, fused))
        rows.append(_metric_row(image_id, "uniform_fusion", clean, uniform))
        if args.save_images:
            stem = f"{image_index:03d}_{Path(image_id).stem}"
            save_image(fused, args.output_dir / "images" / f"{stem}_fused.png")

    summary = {
        "weights_file": str(args.weights.resolve()),
        "test_dataset": str(args.clean_directory.resolve()),
        "noise": {"type": "awgn", "sigma_255": sigma, "clip_noisy_input": False},
        "seed": args.seed,
        "crop_size": args.crop_size,
        "image_count": len(paths),
        "filter_weights": weight_map,
        "metrics": _summary(rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "evaluation_metrics.csv",
        rows,
        ["image", "method", "mse", "mae", "psnr", "ssim"],
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fit":
        fit(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
