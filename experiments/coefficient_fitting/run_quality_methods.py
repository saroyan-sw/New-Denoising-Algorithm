from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from denoise_research.metrics import full_reference_metrics
from denoise_research.nr_metrics import MetricSpec, PyIQACollection
from denoise_research.spectral_ensemble import combine_affinities, metric_to_affinity
from run_experiment import (
    COARSE_SIGMA,
    FINE_SIGMA,
    SIGMA,
    WEIGHT_GRID,
    components,
    crop,
    fuse,
    image_files,
    load_image,
    noisy_pair,
    prepare_objective,
)

METRIC_FAILURES: dict[str, int] = {"brisque": 0, "niqe": 0}


def coefficient_grid() -> np.ndarray:
    return np.stack(np.meshgrid(WEIGHT_GRID, WEIGHT_GRID, WEIGHT_GRID, indexing="ij"), -1).reshape(-1, 3)


def score_batch(scorer: PyIQACollection, images: list[np.ndarray], batch_size: int = 16) -> dict[str, np.ndarray]:
    values = {spec.name: [] for spec in scorer.specs}
    torch = scorer.torch
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            arrays = np.stack([np.clip(image, 0.0, 1.0) for image in images[start:start + batch_size]])
            tensor = torch.from_numpy(np.ascontiguousarray(arrays)).permute(0, 3, 1, 2).to(scorer.device)
            for name, metric in scorer.metrics.items():
                try:
                    output = metric(tensor).detach().reshape(-1).cpu().numpy()
                    if output.size != tensor.shape[0]:
                        raise ValueError(f"Expected {tensor.shape[0]} metric scores, got {output.size}")
                    METRIC_FAILURES[name] += int(np.count_nonzero(~np.isfinite(output)))
                    values[name].extend(float(x) for x in output)
                except Exception:
                    # NIQE can be undefined for nearly constant candidates. Score
                    # the rest individually and record invalid metric outputs.
                    for sample in tensor:
                        try:
                            value = metric(sample.unsqueeze(0)).detach().reshape(-1).cpu().numpy()
                            number = float(value[0]) if value.size == 1 else float("nan")
                        except Exception:
                            number = float("nan")
                        if not np.isfinite(number):
                            METRIC_FAILURES[name] += 1
                        values[name].append(number)
    result = {name: np.asarray(scores, dtype=np.float64) for name, scores in values.items()}
    # Lower is better for BRISQUE/NIQE. A failed score is made worse than every
    # valid score for that image; if all scores fail, stop rather than invent data.
    for name, matrix in result.items():
        rows = [matrix] if matrix.ndim == 1 else list(matrix)
        for row in rows:
            valid = np.isfinite(row)
            if not valid.any():
                raise RuntimeError(f"Every {name} score failed for one image")
            row[~valid] = float(np.max(row[valid]) + 1.0)
    return result


def affinity_matrices(raw: dict[str, np.ndarray], lower: bool = True) -> dict[str, np.ndarray]:
    return {name: metric_to_affinity(scores, lower_is_better=lower, mode="rank")
            for name, scores in raw.items()}


def quality_utility(affinities: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(list(affinities.values()), axis=0), axis=(0, 1))


def choose_global(utility: np.ndarray, sure: np.ndarray, tolerance: float) -> tuple[int, int]:
    quality_index = int(np.argmax(utility))
    safe = np.flatnonzero(sure <= float(np.min(sure)) + tolerance)
    constrained_index = int(safe[np.argmax(utility[safe])])
    return quality_index, constrained_index


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    os.environ["TORCH_HOME"] = str((ROOT / ".cache" / "torch").resolve())
    import torch  # imported after choosing a workspace-local model cache

    scorer = PyIQACollection([MetricSpec("brisque", True), MetricSpec("niqe", True)], device="cpu")
    weights = coefficient_grid()
    train_paths = image_files(ROOT / "data" / "DIV2K_valid")[:args.train_images]
    train_records = []
    train_affinity_by_metric: dict[str, list[np.ndarray]] = {"brisque": [], "niqe": []}
    for index, path in enumerate(train_paths):
        clean, noisy = noisy_pair(path, args.seed + index, args.crop_size)
        bands = components(noisy)
        train_records.append((bands, noisy, None))
        candidate_images = [fuse(bands, w, clip=True) for w in weights]
        raw = score_batch(scorer, candidate_images)
        for name in raw:
            train_affinity_by_metric[name].append(
                metric_to_affinity(raw[name][None, :], lower_is_better=True, mode="rank")[0]
            )

    train_affinities = {name: np.stack(rows) for name, rows in train_affinity_by_metric.items()}
    utility = quality_utility(train_affinities)
    sure_objective = prepare_objective(train_records, "sure")
    sure = np.asarray([sure_objective(w) for w in weights])
    tolerance = args.risk_tolerance_sigma2 * SIGMA**2
    quality_idx, constrained_idx = choose_global(utility, sure, tolerance)
    chosen = {
        "global_quality_only": weights[quality_idx],
        "global_risk_constrained_quality": weights[constrained_idx],
    }

    eval_rows: list[dict] = []
    for dataset in ("CBSD68", "Kodak24"):
        paths = image_files(ROOT / "data" / dataset)[:args.test_images]
        for index, path in enumerate(paths):
            clean, noisy = noisy_pair(path, args.seed + 10000 + index, args.eval_crop_size)
            bands = components(noisy)
            candidates = [fuse(bands, w, clip=True) for w in weights]
            raw = score_batch(scorer, candidates)
            aff = affinity_matrices({name: vals[None, :] for name, vals in raw.items()})
            per_image_utility = np.mean(np.stack([value[0] for value in aff.values()]), axis=0)
            local_sure_objective = prepare_objective([(bands, noisy, None)], "sure")
            local_sure = np.asarray([local_sure_objective(w) for w in weights])
            local_quality = int(np.argmax(per_image_utility))
            local_safe = np.flatnonzero(local_sure <= float(np.min(local_sure)) + tolerance)
            local_constrained = int(local_safe[np.argmax(per_image_utility[local_safe])])

            methods = dict(chosen)
            methods["per_image_quality_only"] = weights[local_quality]
            methods["per_image_risk_constrained_quality"] = weights[local_constrained]
            for method, coefficients in methods.items():
                estimate = fuse(bands, coefficients, clip=True)
                eval_rows.append({
                    "dataset": dataset,
                    "image": path.name,
                    "method": method,
                    "low": float(coefficients[0]),
                    "mid": float(coefficients[1]),
                    "high": float(coefficients[2]),
                    **full_reference_metrics(clean, estimate),
                })

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "nr_per_image_results.csv", eval_rows)
    methods = sorted({row["method"] for row in eval_rows})
    summary = []
    for dataset in ("CBSD68", "Kodak24"):
        for method in methods:
            rows = [row for row in eval_rows if row["dataset"] == dataset and row["method"] == method]
            summary.append({"dataset": dataset, "method": method, "images": len(rows),
                            "mean_psnr": float(np.mean([row["psnr"] for row in rows])),
                            "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
                            "mean_mse": float(np.mean([row["mse"] for row in rows]))})
    write_csv(out / "nr_summary.csv", summary)
    fit_rows = []
    for method, index in (("global_quality_only", quality_idx),
                          ("global_risk_constrained_quality", constrained_idx)):
        fit_rows.append({"method": method, "low": float(weights[index, 0]),
                         "mid": float(weights[index, 1]), "high": float(weights[index, 2]),
                         "mean_sure": float(sure[index]),
                         "mean_quality_affinity": float(utility[index]),
                         "risk_threshold": float(np.min(sure) + tolerance)})
    write_csv(out / "nr_fitted_global_weights.csv", fit_rows)
    (out / "nr_metric_failures.json").write_text(json.dumps(METRIC_FAILURES, indent=2), encoding="utf-8")
    print(f"saved NR coefficient comparison to {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--train-images", type=int, default=16)
    parser.add_argument("--test-images", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--eval-crop-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--risk-tolerance-sigma2", type=float, default=0.1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
