"""Configurable, resumable runs for capacity checks and notebook exploration."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

from checkpoint import load_checkpoint, save_checkpoint
from models import AffineGenerator, make_kan, parameter_count
from state import ROOT, atomic_json_write, load_state, update_state
from train import fixed_validation, restore_rng, rng_state, set_seed, uniform_open, validate


@dataclass
class ExperimentConfig:
    run_id: str
    model: str = "kan"
    widths: list[int] = field(default_factory=lambda: [1, 8, 8, 1])
    grid_intervals: int = 32
    spline_order: int = 3
    activation: str = "silu"
    seed: int = 1729
    batch_size: int = 2048
    steps: int = 1000
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss: str = "w2"
    target_std: float = 0.2
    validation_size: int = 8192
    checkpoint_every: int = 100
    validate_every: int = 25
    device: str = "auto"

    def check(self):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.run_id):
            raise ValueError("run_id must be a simple unique name, without paths")
        if self.model not in {"kan", "mlp", "affine"} or self.loss not in {"w2", "w1"}:
            raise ValueError("unsupported model or loss")
        if self.optimizer not in {"adam", "adamw"} or self.activation not in {"silu", "tanh", "relu"}:
            raise ValueError("unsupported optimizer or activation")
        if len(self.widths) < 2 or self.widths[0] != 1 or self.widths[-1] != 1 or min(self.widths) < 1:
            raise ValueError("widths must start/end at 1 and contain positive integers")
        if min(self.batch_size, self.steps, self.grid_intervals, self.spline_order, self.validation_size, self.checkpoint_every, self.validate_every) < 1:
            raise ValueError("sizes and intervals must be positive")
        if not math.isfinite(self.target_std) or self.target_std <= 0 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("scale and learning rate must be finite and positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("unsupported device")


def build_model(config: ExperimentConfig):
    # Only architecture fields enter the generator; target parameters stay in the sampler.
    if config.model == "kan":
        return make_kan(config.widths, config.grid_intervals, config.spline_order)
    if config.model == "affine":
        return AffineGenerator()
    activation = {"silu": nn.SiLU, "tanh": nn.Tanh, "relu": nn.ReLU}[config.activation]
    layers = []
    for index, (inputs, outputs) in enumerate(zip(config.widths, config.widths[1:])):
        layers.append(nn.Linear(inputs, outputs))
        if index < len(config.widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


def distribution_loss(output, target, kind="w2"):
    difference = output.flatten().sort().values - target.flatten().sort().values
    return difference.square().mean() if kind == "w2" else difference.abs().mean()


def _write_history(path, rows):
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "val_w2_squared", "output_mean", "output_std", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


STOP_REQUESTED = False


def run_session(config: ExperimentConfig, runs_root=None, *, resume=False,
                max_runtime_seconds=600, max_steps_per_session=None, track_study=False):
    """One bounded session; explicit resume is mandatory for any existing run."""
    global STOP_REQUESTED
    STOP_REQUESTED = False
    config.check()
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("runtime budget must be positive")
    if max_steps_per_session is not None and max_steps_per_session <= 0:
        raise ValueError("step budget must be positive")
    run_dir = Path(runs_root or ROOT / "runs").resolve() / config.run_id
    cp_dir = run_dir / "checkpoints"
    state_path, metric_path = run_dir / "state.json", run_dir / "metrics.csv"
    latest = cp_dir / "latest.pt"
    settings = asdict(config)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if config.device != "cpu" and torch.cuda.is_available() else "cpu"
    cp_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (state_path.exists() or latest.exists()):
        raise FileExistsError(f"Run {config.run_id} exists. Resume explicitly or use a new run_id.")
    if resume and not latest.exists():
        raise FileNotFoundError(f"No recovery checkpoint for {config.run_id}; refusing to restart")
    model = None
    history = []
    if resume:
        saved = load_checkpoint(latest, map_location=device)
        if saved["config"] != settings:
            raise ValueError("configuration changed since checkpoint; use a new run_id")
        if saved["step"] == config.steps:
            return json.loads(state_path.read_text(encoding="utf-8"))
        model = build_model(config).to(device)
        model.load_state_dict(saved["model_state"])
        optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config.optimizer](model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"])
        step_done = saved["step"]
        elapsed_before = saved["metadata"]["elapsed_seconds"]
        best_metric, best_name = saved["best_validation_metric"], saved["best_checkpoint"]
        if metric_path.exists():
            with metric_path.open(newline="", encoding="utf-8") as handle:
                history = [row for row in csv.DictReader(handle) if int(row["step"]) <= step_done]
    else:
        atomic_json_write(settings, run_dir / "config.json")
        set_seed(config.seed)
        model = build_model(config).to(device)
        optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[config.optimizer](model.parameters(), lr=config.learning_rate)
        step_done, elapsed_before, best_metric, best_name = 0, 0.0, math.inf, None
    val_data = fixed_validation(config.validation_size, config.target_std, device)
    validation = validate(model, *val_data)
    last_validation_step = step_done
    last_loss = None
    begin = time.perf_counter()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    status = {"run_id": config.run_id, "status": "running", "last_completed_step": step_done,
              "checkpoint_step": step_done, "latest_checkpoint": "checkpoints/latest.pt", "best_checkpoint": best_name,
              "parameter_count": parameter_count(model), "next_action": "Resume latest if interrupted", "error": None}

    def elapsed():
        return elapsed_before + time.perf_counter() - begin

    def payload():
        return {"schema_version": 1, "step": step_done, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": None, "config": settings,
                "rng_state": rng_state(), "validation": validation, "train_metric": last_loss,
                "best_validation_metric": best_metric, "best_checkpoint": best_name,
                "metadata": {"elapsed_seconds": elapsed(), "torch_version": torch.__version__,
                             "device": device, "parameter_count": parameter_count(model), "runner_sha256": runner_hash}}

    def persist(periodic=False):
        save_checkpoint(payload(), latest)
        if periodic:
            save_checkpoint(payload(), cp_dir / f"step_{step_done:06d}.pt")
        _write_history(metric_path, history)
        status.update(last_completed_step=step_done, checkpoint_step=step_done, elapsed_seconds=elapsed(), best_checkpoint=best_name)
        atomic_json_write(status, state_path)
        if track_study:
            update_state(milestone=5, status=status["status"], active_experiment=config.run_id,
                         last_completed_step=step_done, latest_checkpoint=latest.relative_to(ROOT).as_posix(),
                         next_action=f"Resume {config.run_id} using its stored config if interrupted")

    def record_validation():
        nonlocal validation, last_validation_step, best_metric, best_name
        validation = validate(model, *val_data)
        last_validation_step = step_done
        history.append({"step": step_done, "train_loss": "" if last_loss is None else last_loss,
                        **validation, "elapsed_seconds": elapsed()})
        if validation["val_w2_squared"] < best_metric:
            best_metric = validation["val_w2_squared"]
            best_name = f"best_step_{step_done:06d}.pt"
            save_checkpoint(payload(), cp_dir / best_name)

    if not resume:
        record_validation()
        save_checkpoint(payload(), cp_dir / "initial.pt")
    persist()
    limit = min(config.steps, step_done + max_steps_per_session) if max_steps_per_session else config.steps
    print(f"{config.run_id}: {config.model}, {parameter_count(model)} parameters, starting step {step_done}", flush=True)
    try:
        while step_done < limit:
            if STOP_REQUESTED or (max_runtime_seconds and time.perf_counter() - begin >= max_runtime_seconds):
                break
            model.train()
            latent = uniform_open((config.batch_size, 1), device)
            target = torch.randn((config.batch_size, 1), device=device) * config.target_std
            loss = distribution_loss(model(latent), target, config.loss)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("missing or non-finite gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite parameter after optimizer update")
            step_done += 1
            last_loss = float(loss.item())
            if step_done % config.validate_every == 0 or step_done == limit:
                record_validation()
                print(f"  step {step_done}: val W2={math.sqrt(validation['val_w2_squared']):.6f}", flush=True)
            if step_done % config.checkpoint_every == 0:
                persist(periodic=True)
    except KeyboardInterrupt:
        # A notebook interrupt can arrive inside optimizer.step. Preserve the
        # last completed atomic checkpoint rather than saving a partial update.
        recovered = load_checkpoint(latest)
        status.update(status="interrupted", last_completed_step=recovered["step"], checkpoint_step=recovered["step"],
                      next_action="Resume latest; work after the last atomic checkpoint was discarded")
        _write_history(metric_path, [row for row in history if int(row["step"]) <= recovered["step"]])
        atomic_json_write(status, state_path)
        if track_study:
            update_state(status="interrupted", last_completed_step=recovered["step"],
                         next_action=f"Resume {config.run_id} from its last atomic checkpoint")
        return status
    except Exception as error:
        status.update(status="failed", error=repr(error), next_action="Inspect the failure; resume the last valid checkpoint")
        atomic_json_write(status, state_path)
        if track_study:
            update_state(status="failed", next_action=f"Inspect {config.run_id} failure and last valid checkpoint")
        raise
    if last_validation_step != step_done:
        record_validation()
    status.update(status="completed" if step_done == config.steps else "interrupted",
                  next_action="Inspect verified results" if step_done == config.steps else "Resume latest with the same config")
    persist()
    verified = load_checkpoint(latest)
    assert verified["step"] == step_done
    if status["status"] == "completed":
        atomic_json_write({"config": settings, "final_step": step_done, "best_checkpoint": best_name,
                           "best_validation_w2_squared": best_metric, "final_validation": validation,
                           "parameter_count": parameter_count(model), "training_loop_seconds": elapsed()}, run_dir / "summary.json")
        if track_study:
            completed = list(load_state()["completed_experiments"])
            if config.run_id not in completed:
                completed.append(config.run_id)
            update_state(status="planned", active_experiment=None, completed_experiments=completed,
                         pending_experiments=[item for item in load_state()["pending_experiments"] if item != config.run_id],
                         next_action="Continue the next planned robustness run or evaluate all completed runs")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--resume", choices=["latest"])
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-runtime-seconds", type=float, default=600)
    parser.add_argument("--max-steps-per-session", type=int)
    parser.add_argument("--track-study", action="store_true")
    args = parser.parse_args()
    config = ExperimentConfig(**json.loads(args.config.read_text(encoding="utf-8")))
    if args.status:
        print((args.runs_root / config.run_id / "state.json").read_text(encoding="utf-8"))
        return
    def stop(_signum, _frame):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    run_session(config, args.runs_root, resume=bool(args.resume), max_runtime_seconds=args.max_runtime_seconds,
                max_steps_per_session=args.max_steps_per_session, track_study=args.track_study)


if __name__ == "__main__":
    main()
