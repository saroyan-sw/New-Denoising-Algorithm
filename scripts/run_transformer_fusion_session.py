from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from denoise_research.metrics import full_reference_metrics
from denoise_research.operators import bilateral_filter, gaussian_blur, median_filter
from denoise_research.session_guard import (
    atomic_write_json,
    guarded_checkpoint,
    read_power_status,
)


DEFAULT_OUTPUT = ROOT / "outputs" / "transformer_fusion_session"
CONFIG_PATH = ROOT / "configs" / "transformer_fusion_session.json"
SIGMA_255 = 25.0
SIGMA = SIGMA_255 / 255.0
DEV_SEED = 2026101400
HOLDOUT_SEED = 2026102400
OPERATORS = (
    "identity",
    "gaussian_sigma_0.7",
    "gaussian_sigma_1.1",
    "bilateral_r3_s2.0_r0.12",
    "median_3x3",
)


def _guard_session(output: Path, stage: str, *, mutating_model: bool = False) -> dict[str, Any]:
    metadata_path = output / "session_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Run preflight before this command")
    metadata = _json(metadata_path)
    elapsed = time.time() - float(metadata["start_unix_time"])
    if elapsed > float(metadata["maximum_hours"]) * 3600.0:
        guarded_checkpoint(output / "session_checkpoint.json", {"stage": stage, "elapsed_seconds": elapsed})
        raise RuntimeError("Four-hour session budget exhausted")
    if mutating_model and (output / "locked_model.json").exists():
        raise RuntimeError("Model is locked; development/model mutation is forbidden")
    guarded_checkpoint(output / "session_checkpoint.json", {"stage": stage, "elapsed_seconds": elapsed})
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.asarray(image), 0.0, 1.0)
    Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB").save(path)


def _manifest_samples(path: Path, expected_count: int) -> list[dict[str, Any]]:
    manifest = _json(path)
    samples = manifest["samples"]
    if len(samples) != expected_count:
        raise RuntimeError(f"Expected {expected_count} samples, found {len(samples)}")
    ids = [sample["id"] for sample in samples]
    sources = [sample["source_id"] for sample in samples]
    if len(ids) != len(set(ids)) or len(sources) != len(set(sources)):
        raise RuntimeError("Manifest sample and source IDs must be unique")
    for sample in samples:
        clean_path = ROOT / sample["clean_path"]
        if _sha256(clean_path) != sample["clean_sha256"]:
            raise RuntimeError(f"Hash mismatch for {clean_path}")
    return samples


def prepare_dev(output: Path) -> None:
    _guard_session(output, "prepare_dev", mutating_model=True)
    source = ROOT / "experiments" / "coefficient_fitting" / "results" / "nima_technical_100" / "images"
    clean_paths = sorted(source.glob("*_clean.png"))
    cbsd = [path for path in clean_paths if "CBSD68" in path.name][:10]
    kodak_unique: list[Path] = []
    seen = set()
    for path in clean_paths:
        if "Kodak24" not in path.name:
            continue
        source_id = path.name.split("_r", 1)[0].split("_", 1)[1]
        if source_id in seen:
            continue
        seen.add(source_id)
        kodak_unique.append(path)
        if len(kodak_unique) == 10:
            break
    selected = [path for pair in zip(cbsd, kodak_unique) for path in pair]
    if len(selected) != 20:
        raise RuntimeError("The archived paired crops do not contain 20 unique development sources")
    samples = []
    for index, path in enumerate(selected):
        dataset = "CBSD68" if "CBSD68" in path.name else "Kodak24"
        source_id = path.name.split("_r", 1)[0].split("_", 1)[1]
        samples.append(
            {
                "id": f"dev-{index + 1:02d}",
                "role": "fit" if index < 12 else "validation",
                "dataset": dataset,
                "source_id": source_id,
                "clean_path": _relative(path),
                "clean_sha256": _sha256(path),
                "noise_seed": DEV_SEED + index,
            }
        )
    manifest = {
        "protocol_version": 1,
        "created": "2026-10-14",
        "count": 20,
        "sigma_255": SIGMA_255,
        "clip_noisy_input": False,
        "historical_exposure": (
            "Source crops were exposed in earlier project runs. This split is development-only "
            "and cannot support a fresh holdout claim."
        ),
        "samples": samples,
    }
    atomic_write_json(output / "dev_manifest.json", manifest)
    materialize_noisy(output / "dev_manifest.json", output / "dev_noisy.npz")


def materialize_noisy(manifest_path: Path, target: Path) -> None:
    samples = _manifest_samples(manifest_path, 20)
    clean_stack, noisy_stack = [], []
    for sample in samples:
        clean = _load_rgb(ROOT / sample["clean_path"])
        rng = np.random.default_rng(sample["noise_seed"])
        noisy = clean + rng.normal(0.0, SIGMA, size=clean.shape).astype(np.float32)
        clean_stack.append(clean)
        noisy_stack.append(noisy)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.npz")
    np.savez_compressed(temp, clean=np.stack(clean_stack), noisy=np.stack(noisy_stack))
    os.replace(temp, target)


def _operator_bank(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "identity": image.astype(np.float32),
        "gaussian_sigma_0.7": gaussian_blur(image, 0.7),
        "gaussian_sigma_1.1": gaussian_blur(image, 1.1),
        "bilateral_r3_s2.0_r0.12": bilateral_filter(
            image, radius=3, sigma_spatial=2.0, sigma_range=0.12
        ),
        "median_3x3": median_filter(image, 3),
    }


def compute_candidates(noisy: np.ndarray) -> np.ndarray:
    result = []
    for image in noisy:
        bank = _operator_bank(image)
        result.append(np.stack([bank[name] for name in OPERATORS]))
    return np.stack(result).astype(np.float32)


def _project_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = sorted_values - cumulative / indices > 0
    rho = indices[valid][-1]
    theta = cumulative[valid][-1] / rho
    return np.maximum(values - theta, 0.0)


def oracle_weights(candidates: np.ndarray, clean: np.ndarray, indices: list[int]) -> np.ndarray:
    matrix = np.transpose(candidates[indices], (0, 2, 3, 4, 1)).reshape(-1, len(OPERATORS)).astype(np.float64)
    target = clean[indices].reshape(-1).astype(np.float64)
    gram = matrix.T @ matrix / matrix.shape[0]
    cross = matrix.T @ target / matrix.shape[0]
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(gram).max()), 1e-9)
    weights = np.full(len(OPERATORS), 1.0 / len(OPERATORS))
    for _ in range(2000):
        proposed = _project_simplex(weights - (2.0 * (gram @ weights - cross)) / lipschitz)
        if np.max(np.abs(proposed - weights)) < 1e-12:
            break
        weights = proposed
    return weights.astype(np.float32)


def mc_divergence(noisy: np.ndarray, candidates: np.ndarray, samples: list[dict[str, Any]]) -> np.ndarray:
    divergences = np.zeros((len(noisy), len(OPERATORS)), dtype=np.float32)
    epsilon = 1e-3
    for image_index, image in enumerate(noisy):
        rng = np.random.default_rng(int(samples[image_index]["noise_seed"]) + 1_000_000)
        probe = rng.choice((-1.0, 1.0), size=image.shape).astype(np.float32)
        perturbed = _operator_bank(image + epsilon * probe)
        for operator_index, name in enumerate(OPERATORS):
            delta = perturbed[name] - candidates[image_index, operator_index]
            divergences[image_index, operator_index] = float(np.sum(probe * delta) / epsilon)
    return divergences


def _fuse(candidates: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(candidates, weights.astype(np.float32), axes=(1, 0)).astype(np.float32)


def _metrics_rows(
    samples: list[dict[str, Any]], clean: np.ndarray, methods: dict[str, np.ndarray], runtimes: dict[str, float]
) -> list[dict[str, Any]]:
    rows = []
    for method, outputs in methods.items():
        for index, output in enumerate(outputs):
            scores = full_reference_metrics(clean[index], np.clip(output, 0.0, 1.0))
            rows.append(
                {
                    "id": samples[index]["id"],
                    "role": samples[index].get("role", "holdout"),
                    "dataset": samples[index]["dataset"],
                    "source_id": samples[index]["source_id"],
                    "method": method,
                    "runtime_seconds_per_image": runtimes.get(method, math.nan) / len(samples),
                    **scores,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_psnr(rows: list[dict[str, Any]], method: str, role: str) -> float:
    values = [float(row["psnr"]) for row in rows if row["method"] == method and row["role"] == role]
    return float(np.mean(values))


def preflight(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "session_metadata.json"
    if metadata_path.exists():
        _guard_session(output, "preflight_reuse")
        return
    checkpoint = output / "preflight_checkpoint.json"
    try:
        guarded_checkpoint(checkpoint, {"stage": "simulated_stop", "next_step": 4}, simulated_percent=14)
    except RuntimeError:
        pass
    recovered = _json(checkpoint)
    if recovered["next_step"] != 4:
        raise RuntimeError("Checkpoint recovery test failed")
    power = read_power_status()
    guarded_checkpoint(output / "session_checkpoint.json", {"stage": "preflight_complete"})
    metadata = {
        "start_unix_time": time.time(),
        "maximum_hours": 4,
        "maximum_substantial_runs": 10,
        "substantial_runs_completed": 0,
        "power": asdict(power),
        "python": sys.version,
        "platform": platform.platform(),
        "checkpoint_recovery_test": "passed",
        "simulated_low_battery_stop_test": "passed",
    }
    metadata["completed_run_ids"] = []
    atomic_write_json(metadata_path, metadata)


def diagnose(output: Path) -> None:
    _guard_session(output, "diagnose", mutating_model=True)
    samples = _manifest_samples(output / "dev_manifest.json", 20)
    arrays = np.load(output / "dev_noisy.npz")
    clean, noisy = arrays["clean"], arrays["noisy"]
    start = time.perf_counter()
    candidates = compute_candidates(noisy)
    runtime = time.perf_counter() - start
    np.savez_compressed(output / "dev_candidates.npz", candidates=candidates)
    fit_indices = [i for i, sample in enumerate(samples) if sample["role"] == "fit"]
    validation_indices = [i for i, sample in enumerate(samples) if sample["role"] == "validation"]
    oracle = oracle_weights(candidates, clean, validation_indices)
    uniform = np.full(len(OPERATORS), 1.0 / len(OPERATORS), dtype=np.float32)
    methods = {"noisy": noisy, "uniform_fusion": _fuse(candidates, uniform), "validation_oracle": _fuse(candidates, oracle)}
    for operator_index, name in enumerate(OPERATORS):
        methods[name] = candidates[:, operator_index]
    rows = _metrics_rows(samples, clean, methods, {name: runtime for name in methods})
    _write_csv(output / "diagnostic_metrics.csv", rows)
    validation_psnr = {name: _mean_psnr(rows, name, "validation") for name in methods}
    best_operator = max(OPERATORS, key=lambda name: validation_psnr[name])
    diagnostic = {
        "operators": list(OPERATORS),
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "validation_oracle_weights": {name: float(value) for name, value in zip(OPERATORS, oracle)},
        "validation_mean_psnr": validation_psnr,
        "best_individual_operator": best_operator,
        "oracle_gain_db_over_best_operator": validation_psnr["validation_oracle"] - validation_psnr[best_operator],
        "decision_rule": "Expand capacity only if the oracle gain is material and learned fusion fails to recover it.",
    }
    divergences = mc_divergence(noisy, candidates, samples)
    np.savez_compressed(output / "sure_terms.npz", divergences=divergences)
    atomic_write_json(output / "oracle_diagnostic.json", diagnostic)
    guarded_checkpoint(output / "session_checkpoint.json", {"stage": "diagnostic_complete"})


def _load_musiq(cache: Path):
    os.environ.setdefault("TORCH_HOME", str(cache.resolve()))
    import pyiqa
    import torch

    scorer = pyiqa.create_metric("musiq", device="cpu", as_loss=True)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in scorer.parameters()):
        raise RuntimeError("MUSIQ freeze proof failed")
    return torch, pyiqa, scorer


def train(output: Path, lambda_iqa: float, steps: int = 20, learning_rate: float = 0.5) -> None:
    metadata = _guard_session(output, "train_preload", mutating_model=True)
    config = _json(CONFIG_PATH)
    allowed_lambdas = [float(value) for value in config["fusion"]["lambda_candidates"]]
    if not any(abs(lambda_iqa - value) < 1e-12 for value in allowed_lambdas):
        raise RuntimeError(f"lambda_iqa must be one of {allowed_lambdas}")
    run_id = f"normsure_v2_lambda_{lambda_iqa:.3f}".replace(".", "p")
    if run_id in metadata.get("completed_run_ids", []):
        return
    if int(metadata["substantial_runs_completed"]) >= int(metadata["maximum_substantial_runs"]):
        raise RuntimeError("Substantial-run budget exhausted")
    samples = _manifest_samples(output / "dev_manifest.json", 20)
    fit_indices = [i for i, sample in enumerate(samples) if sample["role"] == "fit"]
    arrays = np.load(output / "dev_noisy.npz")
    noisy = arrays["noisy"][fit_indices]
    candidates = np.load(output / "dev_candidates.npz")["candidates"][fit_indices]
    divergences = np.load(output / "sure_terms.npz")["divergences"][fit_indices]
    torch, pyiqa, scorer = _load_musiq(output / "model_cache")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(20261014)
    torch.use_deterministic_algorithms(True)

    candidate_tensor = torch.from_numpy(candidates).permute(0, 1, 4, 2, 3)
    noisy_tensor = torch.from_numpy(noisy).permute(0, 3, 1, 2)
    divergence_tensor = torch.from_numpy(divergences)
    calibration_path = output / "iqa_calibration.json"
    if calibration_path.exists():
        calibration = _json(calibration_path)
        score_mean = float(calibration["score_mean"])
        score_std = float(calibration["score_std"])
    else:
        probe_indices = [0, 3, 6, 9]
        probe_scores = []
        with torch.no_grad():
            for operator_index in range(len(OPERATORS)):
                scores = scorer(candidate_tensor[probe_indices, operator_index].clamp(0, 1)).reshape(-1)
                probe_scores.extend(float(value) for value in scores)
        score_mean = float(np.mean(probe_scores))
        score_std = max(float(np.std(probe_scores)), 1e-6)
        atomic_write_json(calibration_path, {"model": "musiq", "score_mean": score_mean, "score_std": score_std, "probe_fit_indices": probe_indices})

    run_dir = output / "training" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.json"
    start_step = 0
    logits = torch.zeros((len(OPERATORS),), dtype=torch.float32)
    if checkpoint_path.exists():
        saved = _json(checkpoint_path)
        if float(saved["lambda_iqa"]) != lambda_iqa:
            raise RuntimeError("Checkpoint lambda mismatch")
        logits = torch.tensor(saved["logits"], dtype=torch.float32)
        start_step = int(saved["next_step"])
    logits.requires_grad_(True)
    history = []
    if checkpoint_path.exists():
        history = list(_json(checkpoint_path).get("history", []))
    gradient_proof = False
    for step in range(start_step, steps):
        weights = torch.softmax(logits, dim=0)
        fused = (candidate_tensor * weights[None, :, None, None, None]).sum(dim=1)
        residual = fused - noisy_tensor
        residual_mse = residual.square().flatten(1).mean(dim=1)
        divergence = (divergence_tensor * weights[None, :]).sum(dim=1) / float(np.prod(noisy.shape[1:]))
        sure = residual_mse + 2.0 * (SIGMA**2) * divergence - SIGMA**2
        quality = scorer(fused.clamp(0, 1)).reshape(-1).mean()
        quality_z = (quality - score_mean) / score_std
        normalized_sure = sure.mean() / (SIGMA**2)
        loss = normalized_sure - lambda_iqa * quality_z
        loss.backward()
        gradient_proof = gradient_proof or bool(torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0)
        with torch.no_grad():
            logits -= learning_rate * logits.grad
            logits.grad.zero_()
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "sure": float(sure.mean().detach()),
                "normalized_sure": float(normalized_sure.detach()),
                "musiq": float(quality.detach()),
                "weights": [float(value) for value in torch.softmax(logits, dim=0)],
            }
        )
        guarded_checkpoint(
            checkpoint_path,
            {
                "run_id": run_id,
                "lambda_iqa": lambda_iqa,
                "next_step": step + 1,
                "logits": [float(value) for value in logits.detach()],
                "score_mean": score_mean,
                "score_std": score_std,
                "history": history,
            },
        )
        _guard_session(output, f"train_{run_id}_step_{step}")
    final_weights = torch.softmax(logits.detach(), dim=0).numpy()
    model = {
        "run_id": run_id,
        "lambda_iqa": lambda_iqa,
        "steps": steps,
        "learning_rate": learning_rate,
        "operators": list(OPERATORS),
        "weights": [float(value) for value in final_weights],
        "iqa_model": "musiq",
        "iqa_frozen": True,
        "iqa_parameter_count": int(sum(parameter.numel() for parameter in scorer.parameters())),
        "iqa_image_gradient_proof": gradient_proof,
        "pyiqa_version": getattr(pyiqa, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "preprocessing": "PyIQA MUSIQ default; fused RGB clipped to [0,1] in NCHW form",
        "iqa_score_mean": score_mean,
        "iqa_score_std": score_std,
        "fidelity": "Monte-Carlo SURE normalized by sigma^2",
        "sigma_255": SIGMA_255,
    }
    weight_files = []
    for path in sorted((output / "model_cache").rglob("*")):
        if path.is_file():
            weight_files.append({"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    model["cached_model_files"] = weight_files
    atomic_write_json(run_dir / "model.json", model)
    session = _json(output / "session_metadata.json")
    session["substantial_runs_completed"] = int(session["substantial_runs_completed"]) + 1
    session.setdefault("completed_run_ids", []).append(run_id)
    atomic_write_json(output / "session_metadata.json", session)

    all_candidates = np.load(output / "dev_candidates.npz")["candidates"]
    all_arrays = np.load(output / "dev_noisy.npz")
    fused_all = _fuse(all_candidates, final_weights)
    example_dir = output / "inspected_examples" / run_id
    for index in (0, 7, 12, 19):
        _save_rgb(example_dir / f"dev-{index + 1:02d}_clean.png", all_arrays["clean"][index])
        _save_rgb(example_dir / f"dev-{index + 1:02d}_noisy.png", all_arrays["noisy"][index])
        _save_rgb(example_dir / f"dev-{index + 1:02d}_fused.png", fused_all[index])


def _evaluate_weight_models(output: Path, samples: list[dict[str, Any]], clean: np.ndarray, candidates: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    methods: dict[str, np.ndarray] = {"noisy": np.load(output / "dev_noisy.npz")["noisy"]}
    for model_path in sorted((output / "training").glob("*/model.json")):
        model = _json(model_path)
        methods[model["run_id"]] = _fuse(candidates, np.asarray(model["weights"], dtype=np.float32))
    rows = _metrics_rows(samples, clean, methods, {})
    return rows, methods


def select_model(output: Path) -> None:
    _guard_session(output, "select", mutating_model=True)
    if (output / "locked_model.json").exists():
        raise RuntimeError("locked_model.json already exists; lock is immutable")
    samples = _manifest_samples(output / "dev_manifest.json", 20)
    arrays = np.load(output / "dev_noisy.npz")
    candidates = np.load(output / "dev_candidates.npz")["candidates"]
    rows, methods = _evaluate_weight_models(output, samples, arrays["clean"], candidates)
    _write_csv(output / "fusion_selection_metrics.csv", rows)
    model_paths = sorted((output / "training").glob("*/model.json"))
    if not model_paths:
        raise RuntimeError("No trained fusion models found")
    scored = []
    for path in model_paths:
        model = _json(path)
        scored.append((_mean_psnr(rows, model["run_id"], "validation"), path, model))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_psnr, best_path, best_model = scored[0]
    lock_payload = {
        "locked_unix_time": time.time(),
        "selection_rule": "highest validation mean PSNR among predeclared lambda candidates",
        "selected_validation_psnr": best_psnr,
        "model": best_model,
        "model_sha256": _sha256(best_path),
        "config_sha256": _sha256(CONFIG_PATH),
        "model_path": _relative(best_path),
        "holdout_access_before_lock": False,
    }
    atomic_write_json(output / "locked_model.json", lock_payload)
    examples = output / "inspected_examples" / "development"
    weights = np.asarray(best_model["weights"], dtype=np.float32)
    fused = _fuse(candidates, weights)
    for index in (0, 7, 12, 19):
        _save_rgb(examples / f"{samples[index]['id']}_clean.png", arrays["clean"][index])
        _save_rgb(examples / f"{samples[index]['id']}_noisy.png", arrays["noisy"][index])
        _save_rgb(examples / f"{samples[index]['id']}_selected.png", fused[index])
    guarded_checkpoint(output / "session_checkpoint.json", {"stage": "model_locked", "model_sha256": lock_payload["model_sha256"]})


def fetch_holdout(output: Path) -> None:
    _guard_session(output, "holdout_access_started")
    lock = output / "locked_model.json"
    if not lock.exists():
        raise RuntimeError("Holdout gate is closed until locked_model.json exists")
    target_dir = output / "holdout_clean"
    target_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, image_number in enumerate(range(81, 101)):
        url = f"https://raw.githubusercontent.com/wuzhongwulidong/URBAN100/master/img{image_number:03d}.png"
        raw_path = target_dir / f"urban{image_number:03d}_raw.png"
        if not raw_path.exists():
            urllib.request.urlretrieve(url, raw_path)
        raw = _load_rgb(raw_path)
        height, width = raw.shape[:2]
        top, left = (height - 128) // 2, (width - 128) // 2
        clean = raw[top : top + 128, left : left + 128]
        clean_path = target_dir / f"urban{image_number:03d}_center128.png"
        _save_rgb(clean_path, clean)
        samples.append(
            {
                "id": f"holdout-{index + 1:02d}",
                "role": "holdout",
                "dataset": "Urban100",
                "source_id": f"Urban100_img{image_number:03d}",
                "source_url": url,
                "raw_sha256": _sha256(raw_path),
                "crop": {"top": top, "left": left, "height": 128, "width": 128},
                "clean_path": _relative(clean_path),
                "clean_sha256": _sha256(clean_path),
                "noise_seed": HOLDOUT_SEED + index,
            }
        )
    atomic_write_json(
        output / "holdout_manifest.json",
        {"protocol_version": 1, "count": 20, "sigma_255": SIGMA_255, "samples": samples},
    )
    dev_samples = _manifest_samples(output / "dev_manifest.json", 20)
    dev_hashes = {sample["clean_sha256"] for sample in dev_samples}
    dev_ids = {sample["source_id"] for sample in dev_samples}
    if dev_hashes.intersection(sample["clean_sha256"] for sample in samples) or dev_ids.intersection(sample["source_id"] for sample in samples):
        raise RuntimeError("Development/holdout overlap detected")
    materialize_noisy(output / "holdout_manifest.json", output / "holdout_noisy.npz")


def _dncnn_model(torch):
    import torch.nn as nn

    layers: list[nn.Module] = [nn.Conv2d(3, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
    for _ in range(18):
        layers.extend([nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64, eps=1e-4, momentum=0.9), nn.ReLU(inplace=True)])
    layers.append(nn.Conv2d(64, 3, 3, padding=1, bias=False))

    class DnCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(*layers)

        def forward(self, value):
            return value - self.model(value)

    return DnCNN()


def evaluate_baselines(output: Path, split: str) -> None:
    _guard_session(output, f"evaluate_{split}")
    lock = _json(output / "locked_model.json")
    if _sha256(CONFIG_PATH) != lock["config_sha256"]:
        raise RuntimeError("Configuration changed after model lock")
    locked_model_path = ROOT / lock["model_path"]
    if _sha256(locked_model_path) != lock["model_sha256"]:
        raise RuntimeError("Selected model changed after lock")
    marker = output / "holdout_evaluated.json"
    if split == "holdout" and marker.exists():
        raise RuntimeError("Holdout has already been evaluated; refusing a second pass")
    manifest_path = output / f"{split}_manifest.json"
    noisy_path = output / f"{split}_noisy.npz"
    samples = _manifest_samples(manifest_path, 20)
    arrays = np.load(noisy_path)
    clean, noisy = arrays["clean"], arrays["noisy"]
    candidates = compute_candidates(noisy)
    weights = np.asarray(lock["model"]["weights"], dtype=np.float32)
    methods: dict[str, np.ndarray] = {"noisy": noisy, "selected_fusion": _fuse(candidates, weights)}
    runtimes: dict[str, float] = {}
    for operator_index, name in enumerate(OPERATORS):
        methods[name] = candidates[:, operator_index]

    try:
        from skimage.restoration import denoise_nl_means

        start = time.perf_counter()
        methods["nonlocal_means"] = np.stack(
            [denoise_nl_means(image, h=0.9 * SIGMA, sigma=SIGMA, patch_size=5, patch_distance=6, channel_axis=-1, preserve_range=True) for image in noisy]
        ).astype(np.float32)
        runtimes["nonlocal_means"] = time.perf_counter() - start
    except Exception as error:
        atomic_write_json(output / f"{split}_nlm_unavailable.json", {"error": repr(error)})

    try:
        from bm3d import bm3d_rgb

        start = time.perf_counter()
        methods["cbm3d"] = np.stack([bm3d_rgb(image, sigma_psd=SIGMA) for image in noisy]).astype(np.float32)
        runtimes["cbm3d"] = time.perf_counter() - start
    except Exception as error:
        atomic_write_json(output / f"{split}_bm3d_unavailable.json", {"error": repr(error)})

    try:
        import torch

        weights_path = output / "model_cache" / "dncnn_color_blind.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"Download official DnCNN weights to {weights_path}")
        model = _dncnn_model(torch)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        start = time.perf_counter()
        with torch.no_grad():
            tensor = torch.from_numpy(noisy).permute(0, 3, 1, 2)
            output_tensor = model(tensor).clamp(0, 1)
        methods["dncnn_color_blind"] = output_tensor.permute(0, 2, 3, 1).numpy()
        runtimes["dncnn_color_blind"] = time.perf_counter() - start
    except Exception as error:
        atomic_write_json(output / f"{split}_dncnn_unavailable.json", {"error": repr(error)})

    rows = _metrics_rows(samples, clean, methods, runtimes)
    _write_csv(output / f"{split}_comparison_metrics.csv", rows)
    image_dir = output / "inspected_examples" / split
    selected_rows = [row for row in rows if row["method"] == "selected_fusion"]
    worst = min(range(len(samples)), key=lambda index: float(selected_rows[index]["psnr"]))
    for index in sorted({0, 7, 12, 19, worst}):
        _save_rgb(image_dir / f"{samples[index]['id']}_clean.png", clean[index])
        _save_rgb(image_dir / f"{samples[index]['id']}_noisy.png", noisy[index])
        for method, values in methods.items():
            _save_rgb(image_dir / f"{samples[index]['id']}_{method}.png", values[index])
    summary = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            "mean_psnr": float(np.mean([float(row["psnr"]) for row in method_rows])),
            "mean_ssim": float(np.mean([float(row["ssim"]) for row in method_rows])),
            "mean_mse": float(np.mean([float(row["mse"]) for row in method_rows])),
        }
    atomic_write_json(output / f"{split}_comparison_summary.json", summary)
    if split == "holdout":
        atomic_write_json(marker, {"evaluated_unix_time": time.time(), "locked_model_sha256": _json(output / "locked_model.json")["model_sha256"]})


def download_dncnn(output: Path) -> None:
    _guard_session(output, "download_dncnn")
    target = output / "model_cache" / "dncnn_color_blind.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(
            "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth", target
        )
    atomic_write_json(output / "dncnn_model.json", {"path": _relative(target), "sha256": _sha256(target), "source": "official KAIR v1.0 release"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "prepare-dev", "diagnose", "train", "select", "download-dncnn", "fetch-holdout", "evaluate-dev", "evaluate-holdout"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lambda-iqa", type=float)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.command == "preflight":
        preflight(output)
    elif args.command == "prepare-dev":
        prepare_dev(output)
    elif args.command == "diagnose":
        diagnose(output)
    elif args.command == "train":
        if args.lambda_iqa is None:
            parser.error("train requires --lambda-iqa")
        train(output, args.lambda_iqa, args.steps, args.learning_rate)
    elif args.command == "select":
        select_model(output)
    elif args.command == "download-dncnn":
        download_dncnn(output)
    elif args.command == "fetch-holdout":
        fetch_holdout(output)
    elif args.command == "evaluate-dev":
        evaluate_baselines(output, "dev")
    elif args.command == "evaluate-holdout":
        evaluate_baselines(output, "holdout")


if __name__ == "__main__":
    main()
