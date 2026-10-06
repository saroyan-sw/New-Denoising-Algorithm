from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from denoise_research.io import image_files, load_image
from denoise_research.metrics import full_reference_metrics
from denoise_research.pipelines import complementary_bands


WEIGHT_GRID = np.linspace(0.0, 2.0, 5, dtype=np.float32)
SIGMA = 25.0 / 255.0
FINE_SIGMA = 1.0
COARSE_SIGMA = 2.0
ARCHIVE_ROOT = Path(r"C:\Users\user\Downloads\archive_code_only")


def crop(image: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape[:2]
    if min(h, w) <= size:
        return image
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return image[y : y + size, x : x + size]


def noisy_pair(path: Path, seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    clean = crop(load_image(path), size, np.random.default_rng(seed ^ 0x51A7))
    noise = np.random.default_rng(seed).normal(0.0, SIGMA, clean.shape).astype(np.float32)
    # No clipping: classical SURE assumes additive, zero-mean Gaussian noise.
    return clean, clean + noise


def components(noisy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return complementary_bands(noisy, FINE_SIGMA, COARSE_SIGMA)


def fuse(bands: tuple[np.ndarray, ...], weights: np.ndarray, clip: bool = False) -> np.ndarray:
    out = sum(float(weight) * band for weight, band in zip(weights, bands))
    return np.clip(out, 0.0, 1.0) if clip else out.astype(np.float32)


def estimate_sure(bands: tuple[np.ndarray, ...], noisy: np.ndarray, weights: np.ndarray,
                  sigma: float = SIGMA) -> float:
    estimate = fuse(bands, weights)
    residual = estimate - noisy
    # For linear complementary Gaussian bands, these Hutchinson products estimate
    # the normalized divergence. With a Rademacher probe the estimator is stable.
    rng = np.random.default_rng(271828)
    probe = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=noisy.shape)
    probe_bands = complementary_bands(probe, FINE_SIGMA, COARSE_SIGMA)
    divergence = sum(float(w) * float(np.mean(probe * b))
                     for w, b in zip(weights, probe_bands))
    return float(np.mean(residual * residual) + 2.0 * sigma**2 * divergence - sigma**2)


def batch_objective(records: list[tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray | None]],
                    weights: np.ndarray, kind: str) -> float:
    values = []
    for bands, noisy, clean in records:
        if kind == "sure":
            values.append(estimate_sure(bands, noisy, weights))
        else:
            assert clean is not None
            values.append(float(np.mean((fuse(bands, weights) - clean) ** 2)))
    return float(np.mean(values))


def prepare_objective(records: list[tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray | None]],
                      kind: str) -> Callable[[np.ndarray], float]:
    """Precompute quadratic sufficient statistics for fast coefficient searches."""
    stats = []
    for bands, noisy, clean in records:
        matrix = np.stack(bands).reshape(len(bands), -1).astype(np.float64)
        obs = noisy.reshape(-1).astype(np.float64)
        gram = matrix @ matrix.T / matrix.shape[1]
        if kind == "sure":
            probe = np.random.default_rng(271828).choice(
                np.array([-1.0, 1.0], dtype=np.float32), size=noisy.shape
            )
            probe_bands = np.stack(complementary_bands(probe, FINE_SIGMA, COARSE_SIGMA)).reshape(len(bands), -1)
            divergence = np.mean(probe.reshape(-1)[None, :] * probe_bands, axis=1)
            linear = matrix @ obs / matrix.shape[1]
            constant = float(np.mean(obs * obs) - SIGMA**2)
            stats.append((gram, linear, constant, divergence.astype(np.float64)))
        else:
            assert clean is not None
            target = clean.reshape(-1).astype(np.float64)
            linear = matrix @ target / matrix.shape[1]
            constant = float(np.mean(target * target))
            stats.append((gram, linear, constant, None))

    def objective(weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=np.float64)
        scores = []
        for gram, linear, constant, divergence in stats:
            value = float(w @ gram @ w - 2.0 * w @ linear + constant)
            if kind == "sure":
                value += float(2.0 * SIGMA**2 * (divergence @ w))
            scores.append(value)
        return float(np.mean(scores))
    return objective


def grid_search(objective: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
    candidates = np.stack(np.meshgrid(WEIGHT_GRID, WEIGHT_GRID, WEIGHT_GRID, indexing="ij"), -1).reshape(-1, 3)
    scores = np.asarray([objective(w) for w in candidates])
    best = int(np.argmin(scores))
    return candidates[best], float(scores[best])


def random_search(objective: Callable[[np.ndarray], float], seed: int,
                  trials: int = 512) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    candidates = rng.uniform(0.0, 2.0, size=(trials, 3)).astype(np.float32)
    candidates[0] = 1.0
    scores = np.asarray([objective(w) for w in candidates])
    best = int(np.argmin(scores))
    return candidates[best], float(scores[best])


def coordinate_descent(objective: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
    weights = np.ones(3, dtype=np.float32)
    score = objective(weights)
    for step in (0.5, 0.25, 0.125, 0.0625):
        for _ in range(3):
            changed = False
            for axis in range(3):
                choices = []
                for delta in (-step, 0.0, step):
                    candidate = weights.copy()
                    candidate[axis] = np.clip(candidate[axis] + delta, 0.0, 2.0)
                    choices.append((objective(candidate), candidate))
                candidate_score, candidate = min(choices, key=lambda row: row[0])
                if candidate_score < score:
                    weights, score, changed = candidate, candidate_score, True
            if not changed:
                break
    return weights, float(score)


def finite_difference_gradient(objective: Callable[[np.ndarray], float], weights: np.ndarray,
                               epsilon: float = 1e-3) -> np.ndarray:
    grad = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        plus, minus = weights.copy(), weights.copy()
        plus[axis] = min(2.0, plus[axis] + epsilon)
        minus[axis] = max(0.0, minus[axis] - epsilon)
        denom = float(plus[axis] - minus[axis])
        grad[axis] = (objective(plus) - objective(minus)) / denom if denom else 0.0
    return grad


def gradient_descent(objective: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
    weights = np.ones(3, dtype=np.float32)
    best, best_score = weights.copy(), objective(weights)
    first, second = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    for iteration in range(1, 151):
        grad = finite_difference_gradient(objective, weights)
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad * grad
        corrected_first = first / (1.0 - 0.9**iteration)
        corrected_second = second / (1.0 - 0.999**iteration)
        weights = np.clip(weights - 0.04 * corrected_first / (np.sqrt(corrected_second) + 1e-8), 0.0, 2.0)
        score = objective(weights)
        if score < best_score:
            best, best_score = weights.copy(), score
    return best, float(best_score)


def gaussian_process_search(objective: Callable[[np.ndarray], float], seed: int,
                            trials: int = 48) -> tuple[np.ndarray, float]:
    """Small GP-UCB search; avoids optional Bayesian-optimization packages."""
    rng = np.random.default_rng(seed)
    points = [np.ones(3, dtype=np.float32)]
    scores = [objective(points[0])]
    for _ in range(trials - 1):
        if len(points) < 8:
            candidate = rng.uniform(0.0, 2.0, 3).astype(np.float32)
        else:
            x = np.stack(points)
            y = np.asarray(scores, dtype=np.float64)
            y = (y - y.mean()) / max(y.std(), 1e-12)
            length = 0.65
            distances = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
            kernel = np.exp(-0.5 * distances / (length * length)) + 1e-6 * np.eye(len(x))
            alpha = np.linalg.solve(kernel, y)
            pool = rng.uniform(0.0, 2.0, size=(256, 3))
            cross = np.exp(-0.5 * ((pool[:, None, :] - x[None, :, :]) ** 2).sum(axis=2) / (length * length))
            mean = cross @ alpha
            variance = np.maximum(1.0 - np.sum(cross * np.linalg.solve(kernel, cross.T).T, axis=1), 1e-12)
            candidate = pool[int(np.argmin(mean - 1.5 * np.sqrt(variance)))].astype(np.float32)
        points.append(candidate)
        scores.append(objective(candidate))
    best = int(np.argmin(scores))
    return points[best], float(scores[best])


def cma_es_search(objective: Callable[[np.ndarray], float], seed: int,
                  generations: int = 24, population: int = 12) -> tuple[np.ndarray, float]:
    """Compact covariance-adapting evolution strategy for a three-parameter fit."""
    rng = np.random.default_rng(seed)
    mean = np.ones(3, dtype=np.float64)
    covariance = np.eye(3, dtype=np.float64) * 0.25
    best, best_score = mean.astype(np.float32), objective(mean.astype(np.float32))
    for _ in range(generations):
        samples = rng.multivariate_normal(mean, covariance, size=population)
        samples = np.clip(samples, 0.0, 2.0)
        scores = np.asarray([objective(p.astype(np.float32)) for p in samples])
        order = np.argsort(scores)
        elites = samples[order[: max(2, population // 3)]]
        if scores[order[0]] < best_score:
            best, best_score = samples[order[0]].astype(np.float32), float(scores[order[0]])
        old_mean = mean.copy()
        mean = 0.65 * mean + 0.35 * elites.mean(axis=0)
        centered = elites - old_mean
        covariance = 0.55 * covariance + 0.45 * (centered.T @ centered / len(elites)) + 1e-4 * np.eye(3)
    return best, float(best_score)


SEARCHES: dict[str, Callable] = {
    "grid": lambda f, seed: grid_search(f),
    "random": lambda f, seed: random_search(f, seed),
    "coordinate": lambda f, seed: coordinate_descent(f),
    "gradient": lambda f, seed: gradient_descent(f),
    "bayesian_gp_ucb": lambda f, seed: gaussian_process_search(f, seed),
    "cma_es": lambda f, seed: cma_es_search(f, seed),
}


def fit_patch(bands: tuple[np.ndarray, ...], noisy: np.ndarray, clean: np.ndarray | None,
              objective_name: str) -> np.ndarray:
    record = [(bands, noisy, clean)]
    objective = lambda w: batch_objective(record, w, objective_name)
    return grid_search(objective)[0]


def add_result(rows: list[dict], dataset: str, image_id: str, method: str,
               clean: np.ndarray, restored: np.ndarray, weights: np.ndarray | None) -> None:
    metrics = full_reference_metrics(clean, np.clip(restored, 0.0, 1.0))
    rows.append({"dataset": dataset, "image": image_id, "method": method,
                 "low": None if weights is None else float(weights[0]),
                 "mid": None if weights is None else float(weights[1]),
                 "high": None if weights is None else float(weights[2]), **metrics})


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    train_paths = image_files(ROOT / "data" / "DIV2K_valid")[: args.train_images]
    if not train_paths:
        raise FileNotFoundError("No calibration images found under data/DIV2K_valid")
    train_records = []
    for index, path in enumerate(train_paths):
        clean, noisy = noisy_pair(path, args.seed + index, args.crop_size)
        train_records.append((components(noisy), noisy, clean))

    fitted: dict[tuple[str, str], np.ndarray] = {}
    search_rows = []
    for objective_name in ("oracle_mse", "sure"):
        objective_kind = "sure" if objective_name == "sure" else "oracle_mse"
        objective = prepare_objective(train_records, objective_kind)
        for search_name, search in SEARCHES.items():
            weights, score = search(objective, args.seed + len(search_rows))
            fitted[(objective_name, search_name)] = weights
            search_rows.append({"objective": objective_name, "search": search_name,
                                "low": float(weights[0]), "mid": float(weights[1]),
                                "high": float(weights[2]), "training_objective": score})

    # Search results are saved separately so method selection on the test set is visible.
    write_rows(output / "search_comparison.csv", search_rows)
    eval_rows: list[dict] = []
    methods = [(f"global_{objective}_{search}", weights)
               for (objective, search), weights in fitted.items()]
    for dataset in ("CBSD68", "Kodak24"):
        paths = image_files(ROOT / "data" / dataset)[: args.test_images]
        for index, path in enumerate(paths):
            clean, noisy = noisy_pair(path, args.seed + 10000 + index, args.eval_crop_size)
            bands = components(noisy)
            add_result(eval_rows, dataset, path.name, "noisy", clean, noisy, None)
            for method, weights in methods:
                add_result(eval_rows, dataset, path.name, method, clean, fuse(bands, weights), weights)

            # Test-time per-image fits. Oracle is an upper-bound diagnostic;
            # SURE is the available no-clean-target estimator.
            for objective_name in ("oracle_mse", "sure"):
                objective = prepare_objective([(bands, noisy, clean if objective_name == "oracle_mse" else None)], objective_name)
                for search_name in ("grid", "coordinate", "gradient"):
                    weights, _ = SEARCHES[search_name](objective, args.seed + index)
                    add_result(eval_rows, dataset, path.name,
                               f"per_image_{objective_name}_{search_name}", clean, fuse(bands, weights), weights)

            # Patchwise fits, using 96x96 non-overlapping patches on the fixed evaluation crop.
            patch_size = min(args.patch_size, clean.shape[0], clean.shape[1])
            patch_size -= patch_size % 2
            output_arrays = {"sure": np.empty_like(clean), "oracle_mse": np.empty_like(clean)}
            for y0 in range(0, clean.shape[0], patch_size):
                for x0 in range(0, clean.shape[1], patch_size):
                    y1, x1 = min(y0 + patch_size, clean.shape[0]), min(x0 + patch_size, clean.shape[1])
                    if y1 - y0 < 16 or x1 - x0 < 16:
                        continue
                    nb = tuple(b[y0:y1, x0:x1] for b in bands)
                    ny, cl = noisy[y0:y1, x0:x1], clean[y0:y1, x0:x1]
                    for objective_name in ("sure", "oracle_mse"):
                        w = fit_patch(nb, ny, cl if objective_name == "oracle_mse" else None, objective_name)
                        output_arrays[objective_name][y0:y1, x0:x1] = fuse(nb, w)
            for objective_name, restored in output_arrays.items():
                add_result(eval_rows, dataset, path.name, f"patchwise_{objective_name}_grid", clean, restored, None)

    write_rows(output / "per_image_results.csv", eval_rows)
    summary = summarize(eval_rows)
    write_rows(output / "summary.csv", summary)
    checkpoint_paths = [
        ARCHIVE_ROOT / "cyclegan_output" / "resnet" / "checkpoints" / "netD_Clean_epoch_50.pth",
        ARCHIVE_ROOT / "cyclegan_reproducible_output" / "resnet" / "checkpoints" / "netD_Clean_epoch_100.pth",
    ]
    existing_checkpoints = [str(path) for path in checkpoint_paths if path.is_file()]
    has_torch = importlib.util.find_spec("torch") is not None
    has_pyiqa = importlib.util.find_spec("pyiqa") is not None
    availability = {
        "brisque_niqe_quality_only": "available" if has_torch and has_pyiqa else "not run: PyTorch and/or PyIQA are not installed",
        "risk_constrained_quality": "not run: requires BRISQUE/NIQE (PyIQA) or a calibrated frozen discriminator",
        "discriminator_guided_coefficients": "available to run" if has_torch and existing_checkpoints else "not run: PyTorch is missing or no archived checkpoint was found",
        "discriminator_checkpoints_found_outside_project": existing_checkpoints,
        "hits_filter_ensemble": "not run: its candidate-bank affinities require no-reference metric scores",
        "bayesian_search": "ran: compact Gaussian-process UCB search",
        "evolutionary_search": "ran: compact covariance-adapting evolution strategy",
        "noise_model": "synthetic unclipped AWGN, sigma=25/255; metrics clip restored estimate to display range",
        "calibration": f"{len(train_paths)} DIV2K validation images, random {args.crop_size}px crops",
        "evaluation": f"first {args.test_images} CBSD68 and Kodak24 images, random {args.eval_crop_size}px crops; pilot subset",
        "seed": args.seed,
    }
    (output / "method_status.json").write_text(json.dumps(availability, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in keys:
        selected = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
        result.append({"dataset": dataset, "method": method,
                       "images": len(selected),
                       "mean_psnr": float(np.mean([r["psnr"] for r in selected])),
                       "mean_ssim": float(np.mean([r["ssim"] for r in selected])),
                       "mean_mse": float(np.mean([r["mse"] for r in selected]))})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare low/mid/high coefficient fitting methods.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--train-images", type=int, default=16)
    parser.add_argument("--test-images", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--eval-crop-size", type=int, default=192)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260930)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
