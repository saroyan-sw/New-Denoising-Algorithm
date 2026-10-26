"""Train matched MLP and affine generators under the KAN sampling protocol."""

import argparse
import csv
import math
import os
import signal
import time
from pathlib import Path

import torch

from checkpoint import load_checkpoint, save_checkpoint
from models import AffineGenerator, make_mlp, parameter_count
from state import ROOT, load_state, update_state
from train import (
    checkpoint_payload,
    fixed_validation,
    load_config,
    restore_rng,
    set_seed,
    uniform_open,
    validate,
    wasserstein2_squared,
)


STOP = False


def _stop(_signum, _frame):
    global STOP
    STOP = True


def make_model(kind: str):
    return make_mlp() if kind == "mlp" else AffineGenerator()


def write_metric(path: Path, step: int, train_loss, validation: dict, elapsed: float) -> None:
    fields = ["step", "train_w2_squared", "val_w2_squared", "output_mean", "output_std", "elapsed_seconds"]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({"step": step, "train_w2_squared": train_loss, **validation, "elapsed_seconds": elapsed})
        handle.flush()
        os.fsync(handle.fileno())


def trim_metrics(path: Path, step: int) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or int(rows[-1]["step"]) <= step:
        return
    temp = path.with_suffix(".recover.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(row for row in rows if int(row["step"]) <= step)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def run(kind: str, resume: bool, runtime_limit: float | None, step_limit: int | None) -> None:
    cfg = dict(load_config())
    cfg["model"] = kind
    cfg["experiment_id"] = f"{'mlp_matched' if kind == 'mlp' else 'affine'}_seed_{cfg['seed']}"
    directory = ROOT / "checkpoints" / kind
    directory.mkdir(parents=True, exist_ok=True)
    metric_path = ROOT / "results" / "metrics" / f"{kind}_seed_{cfg['seed']}.csv"
    latest = directory / "latest.pt"
    device = "cuda" if torch.cuda.is_available() and cfg["device"] == "auto" else "cpu"
    if resume:
        saved = load_checkpoint(latest, map_location=device)
        if saved["config"] != cfg:
            raise ValueError("configuration differs from saved run")
        model = make_model(kind).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        restore_rng(saved["rng_state"])
        start = saved["step"]
        if start >= cfg["steps"]:
            raise RuntimeError("this baseline is already complete")
        best_metric = saved["best_validation_metric"]
        best_path = Path(saved["best_checkpoint"]) if saved["best_checkpoint"] else None
        if best_path is not None and not best_path.is_absolute():
            best_path = ROOT / best_path
        validation = saved["validation"]
        elapsed_before = saved["metadata"]["elapsed_seconds"]
        trim_metrics(metric_path, start)
        update_state(last_completed_step=start, latest_checkpoint=str(latest.relative_to(ROOT)))
        print(f"Resumed {cfg['experiment_id']} at step {start}", flush=True)
    else:
        if latest.exists() or load_state()["active_experiment"] is not None:
            raise RuntimeError("existing active run/checkpoint found; resume instead of restarting")
        set_seed(cfg["seed"])
        model = make_model(kind).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        start = 0
        best_metric = math.inf
        best_path = None
        elapsed_before = 0.0
        val_data = fixed_validation(cfg["validation_size"], cfg["target_std"], device)
        validation = validate(model, *val_data)
        write_metric(metric_path, 0, "", validation, 0.0)
        initial = checkpoint_payload(model, optimizer, cfg, 0, None, validation, best_metric, best_path, 0.0)
        save_checkpoint(initial, latest)
        save_checkpoint(initial, directory / "initial.pt")
        update_state(last_completed_step=0, latest_checkpoint=str(latest.relative_to(ROOT)))
        print(f"Started {cfg['experiment_id']} with {parameter_count(model)} parameters", flush=True)
    val_data = fixed_validation(cfg["validation_size"], cfg["target_std"], device)
    update_state(milestone=3, status="running", active_experiment=cfg["experiment_id"], next_action=f"Resume {cfg['experiment_id']} if interrupted")
    begin = time.perf_counter()
    deadline = begin + runtime_limit if runtime_limit else None
    end = min(cfg["steps"], start + step_limit) if step_limit else cfg["steps"]
    step_done = start
    val_step = start
    last_loss = None
    try:
        for step in range(start + 1, end + 1):
            if STOP or (deadline and time.perf_counter() >= deadline):
                break
            model.train()
            latent = uniform_open((cfg["batch_size"], 1), device)
            target = torch.randn((cfg["batch_size"], 1), device=device) * cfg["target_std"]
            output = model(latent)
            loss = wasserstein2_squared(output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"invalid loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f"invalid gradient at step {step}")
            optimizer.step()
            step_done = step
            last_loss = float(loss.item())
            elapsed = elapsed_before + time.perf_counter() - begin
            if step % cfg["validate_every"] == 0 or step == end:
                validation = validate(model, *val_data)
                val_step = step
                write_metric(metric_path, step, last_loss, validation, elapsed)
                if validation["val_w2_squared"] < best_metric:
                    best_metric = validation["val_w2_squared"]
                    best_path = directory / f"best_step_{step:06d}.pt"
                    save_checkpoint(checkpoint_payload(model, optimizer, cfg, step, last_loss, validation, best_metric, best_path, elapsed), best_path)
                print(f"{kind} step={step} val_w2sq={validation['val_w2_squared']:.6g}", flush=True)
            if step % cfg["checkpoint_every"] == 0:
                save_checkpoint(checkpoint_payload(model, optimizer, cfg, step, last_loss, validation, best_metric, best_path, elapsed), latest)
                save_checkpoint(load_checkpoint(latest), directory / f"step_{step:06d}.pt")
                update_state(last_completed_step=step, latest_checkpoint=str(latest.relative_to(ROOT)))
        elapsed = elapsed_before + time.perf_counter() - begin
        if step_done != start and step_done != val_step:
            validation = validate(model, *val_data)
            write_metric(metric_path, step_done, last_loss, validation, elapsed)
        final = checkpoint_payload(model, optimizer, cfg, step_done, last_loss, validation, best_metric, best_path, elapsed)
        save_checkpoint(final, latest)
        update_state(last_completed_step=step_done, latest_checkpoint=str(latest.relative_to(ROOT)))
        if step_done == cfg["steps"]:
            verified = load_checkpoint(latest)
            if verified["step"] != cfg["steps"] or best_path is None or not best_path.is_file():
                raise RuntimeError("baseline completion checkpoint verification failed")
            completed = list(load_state()["completed_experiments"])
            completed.append(cfg["experiment_id"])
            pending = [x for x in load_state()["pending_experiments"] if x != cfg["experiment_id"]]
            update_state(status="planned", active_experiment=None, completed_experiments=completed, pending_experiments=pending, next_action="Run the next baseline or compare completed models")
            print(f"Completed {cfg['experiment_id']}", flush=True)
        else:
            update_state(status="interrupted", next_action=f"python baseline_train.py --model {kind} --resume latest")
            print(f"Interrupted {kind} at step {step_done}", flush=True)
    except Exception:
        update_state(status="failed", next_action=f"Inspect {kind} failure and resume latest valid checkpoint")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["mlp", "affine"], required=True)
    parser.add_argument("--resume", choices=["latest"])
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--max-steps-per-session", type=int)
    args = parser.parse_args()
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("runtime limit must be positive")
    if args.max_steps_per_session is not None and args.max_steps_per_session <= 0:
        parser.error("step limit must be positive")
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _stop)
    run(args.model, bool(args.resume), args.max_runtime_seconds, args.max_steps_per_session)


if __name__ == "__main__":
    main()
