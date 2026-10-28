"""Train a spline KAN from independent uniform and Gaussian sample sets."""

import argparse
import csv
import math
import os
import random
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from checkpoint import load_checkpoint, save_checkpoint
from models import make_kan, parameter_count
from state import ROOT, load_state, status_text, update_state


CONFIG_PATH = ROOT / "config" / "experiment.yaml"
CHECKPOINT_DIR = ROOT / "checkpoints"
METRIC_PATH = ROOT / "results" / "metrics" / "kan_primary_seed_1729.csv"
STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; saving after the current step", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def uniform_open(shape: tuple[int, ...], device, generator=None) -> torch.Tensor:
    values = torch.rand(shape, device=device, generator=generator)
    eps = torch.finfo(values.dtype).eps
    return values.clamp(eps, 1.0 - eps)


def wasserstein2_squared(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if generated.shape != target.shape:
        raise ValueError("generated and target sample shapes differ")
    return (torch.sort(generated.flatten()).values - torch.sort(target.flatten()).values).square().mean()


def fixed_validation(size: int, std: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    latent_rng = torch.Generator(device="cpu").manual_seed(917_271)
    target_rng = torch.Generator(device="cpu").manual_seed(638_219)
    latent = uniform_open((size, 1), "cpu", latent_rng).to(device)
    target = (torch.randn((size, 1), generator=target_rng) * std).to(device)
    return latent, target


@torch.no_grad()
def validate(model, latent: torch.Tensor, target: torch.Tensor) -> dict:
    model.eval()
    output = model(latent)
    if not torch.isfinite(output).all():
        raise FloatingPointError("non-finite validation output")
    return {
        "val_w2_squared": float(wasserstein2_squared(output, target).item()),
        "output_mean": float(output.mean().item()),
        "output_std": float(output.std(unbiased=False).item()),
    }


def append_metric(row: dict) -> None:
    METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "train_w2_squared", "val_w2_squared", "output_mean", "output_std", "grad_norm", "elapsed_seconds"]
    exists = METRIC_PATH.exists() and METRIC_PATH.stat().st_size > 0
    with METRIC_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())


def reconcile_metrics(checkpoint_step: int) -> None:
    """Drop records newer than the recovery point before replaying steps."""
    if not METRIC_PATH.exists():
        return
    with METRIC_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    retained = [row for row in rows if int(row["step"]) <= checkpoint_step]
    if len(retained) == len(rows):
        return
    temporary = METRIC_PATH.with_suffix(".recover.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(retained)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, METRIC_PATH)


def checkpoint_payload(model, optimizer, config, step, train_metric, validation, best_metric, best_path, elapsed) -> dict:
    return {
        "schema_version": 1,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None,
        "config": config,
        "rng_state": rng_state(),
        "train_metric": train_metric,
        "validation": validation,
        "best_validation_metric": best_metric,
        "best_checkpoint": str(best_path.relative_to(ROOT)) if best_path else None,
        "metadata": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "torch_version": torch.__version__,
            "parameter_count": parameter_count(model),
            "device": str(next(model.parameters()).device),
        },
    }


def save_recovery(payload: dict, periodic: bool = False) -> Path:
    latest = CHECKPOINT_DIR / "latest.pt"
    save_checkpoint(payload, latest)
    if periodic:
        save_checkpoint(payload, CHECKPOINT_DIR / f"step_{payload['step']:06d}.pt")
    update_state(last_completed_step=payload["step"], latest_checkpoint=str(latest.relative_to(ROOT)))
    return latest


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["model"] != "kan" or config["batch_size"] < 2 or config["steps"] < 1:
        raise ValueError("invalid primary KAN configuration")
    return config


def train(args) -> None:
    config = load_config()
    device = "cuda" if config["device"] == "auto" and torch.cuda.is_available() else "cpu"
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = "cuda"
    latest = CHECKPOINT_DIR / "latest.pt"
    if args.resume:
        payload = load_checkpoint(latest, map_location=device)
        if payload["config"] != config:
            raise ValueError("configuration differs from recovery checkpoint")
        if payload["step"] >= config["steps"]:
            print(f"{config['experiment_id']} is already complete at step {payload['step']}; no checkpoint was changed")
            return
        model = make_kan(config["widths"], config["grid_intervals"], config["spline_order"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        restore_rng(payload["rng_state"])
        start_step = payload["step"]
        best_metric = payload["best_validation_metric"]
        best_path = Path(payload["best_checkpoint"]) if payload["best_checkpoint"] else None
        if best_path is not None and not best_path.is_absolute():
            best_path = ROOT / best_path
        last_validation = payload["validation"]
        elapsed_before = payload["metadata"]["elapsed_seconds"]
        reconcile_metrics(start_step)
        print(f"Resumed {config['experiment_id']} from step {start_step}", flush=True)
    else:
        if load_state()["active_experiment"] is not None or latest.exists():
            raise RuntimeError("existing run found; use --resume latest, never silently restart")
        set_seed(config["seed"])
        model = make_kan(config["widths"], config["grid_intervals"], config["spline_order"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        start_step = 0
        best_metric = math.inf
        best_path = None
        elapsed_before = 0.0
        val_latent, val_target = fixed_validation(config["validation_size"], config["target_std"], device)
        last_validation = validate(model, val_latent, val_target)
        append_metric({"step": 0, **last_validation, "elapsed_seconds": 0})
        payload = checkpoint_payload(model, optimizer, config, 0, None, last_validation, best_metric, best_path, 0.0)
        save_recovery(payload)
        save_checkpoint(payload, CHECKPOINT_DIR / "initial.pt")
        print(f"Started {config['experiment_id']} with {parameter_count(model)} parameters on {device}", flush=True)
    val_latent, val_target = fixed_validation(config["validation_size"], config["target_std"], device)
    current_milestone = load_state()["milestone"]
    update_state(milestone=max(current_milestone, 1), status="running", active_experiment=config["experiment_id"], next_action="Continue primary KAN training or resume latest after interruption")
    begin = time.perf_counter()
    deadline = begin + args.max_runtime_seconds if args.max_runtime_seconds else None
    end_step = min(config["steps"], start_step + args.max_steps_per_session) if args.max_steps_per_session else config["steps"]
    last_step = start_step
    last_train = None
    last_validation_step = start_step
    try:
        for step in range(start_step + 1, end_step + 1):
            if STOP_REQUESTED or (deadline and time.perf_counter() >= deadline):
                break
            model.train()
            latent = uniform_open((config["batch_size"], 1), device)
            target = torch.randn((config["batch_size"], 1), device=device) * config["target_std"]
            output = model(latent)
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"non-finite generator output at step {step}")
            loss = wasserstein2_squared(output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f"invalid gradients at step {step}")
            grad_norm = math.sqrt(sum(float(g.square().sum().item()) for g in gradients))
            optimizer.step()
            last_step = step
            last_train = float(loss.item())
            elapsed = elapsed_before + time.perf_counter() - begin
            if step % config["validate_every"] == 0 or step == end_step or step == config["steps"]:
                last_validation = validate(model, val_latent, val_target)
                last_validation_step = step
                append_metric({"step": step, "train_w2_squared": last_train, **last_validation, "grad_norm": grad_norm, "elapsed_seconds": elapsed})
                if last_validation["val_w2_squared"] < best_metric:
                    best_metric = last_validation["val_w2_squared"]
                    best_path = CHECKPOINT_DIR / f"best_step_{step:06d}.pt"
                    best_payload = checkpoint_payload(model, optimizer, config, step, last_train, last_validation, best_metric, best_path, elapsed)
                    save_checkpoint(best_payload, best_path)
                print(f"step={step} train_w2sq={last_train:.6g} val_w2sq={last_validation['val_w2_squared']:.6g} output_std={last_validation['output_std']:.5f}", flush=True)
            if step % config["checkpoint_every"] == 0:
                payload = checkpoint_payload(model, optimizer, config, step, last_train, last_validation, best_metric, best_path, elapsed)
                save_recovery(payload, periodic=True)
        elapsed = elapsed_before + time.perf_counter() - begin
        if last_step != start_step and last_step != last_validation_step:
            last_validation = validate(model, val_latent, val_target)
            append_metric({"step": last_step, "train_w2_squared": last_train, **last_validation, "elapsed_seconds": elapsed})
        payload = checkpoint_payload(model, optimizer, config, last_step, last_train, last_validation, best_metric, best_path, elapsed)
        save_recovery(payload, periodic=last_step > 0 and last_step % config["checkpoint_every"] == 0)
        if last_step == config["steps"]:
            completed = list(load_state()["completed_experiments"])
            if config["experiment_id"] not in completed:
                completed.append(config["experiment_id"])
            pending = [item for item in load_state()["pending_experiments"] if item != config["experiment_id"]]
            update_state(milestone=2, status="completed", active_experiment=None, completed_experiments=completed, pending_experiments=pending, next_action="Verify primary run metrics and evaluate baselines")
            print("Training completed and final checkpoint saved", flush=True)
        else:
            update_state(status="interrupted", next_action="python train.py --resume latest --max-runtime-seconds 600")
            print(f"Stopped at step {last_step}; resume from checkpoints/latest.pt", flush=True)
    except Exception as error:
        update_state(status="failed", next_action="Inspect error, latest valid checkpoint, and metric log; then resume latest")
        print(f"Training failed after step {last_step}: {error}", flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show persistent study state")
    parser.add_argument("--resume", choices=["latest"], help="resume a prior run")
    parser.add_argument("--max-runtime-seconds", type=float, default=None)
    parser.add_argument("--max-steps-per-session", type=int, default=None)
    args = parser.parse_args()
    if args.status:
        print(status_text())
        checkpoint_name = load_state().get("latest_checkpoint")
        if checkpoint_name:
            print(f"Checkpoint exists: {(ROOT / checkpoint_name).is_file()}")
        return
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be positive")
    if args.max_steps_per_session is not None and args.max_steps_per_session <= 0:
        parser.error("--max-steps-per-session must be positive")
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _request_stop)
    train(args)


if __name__ == "__main__":
    main()
