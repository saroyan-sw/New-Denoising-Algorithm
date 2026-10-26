"""Atomic persistent experiment state, independent of the training process."""

import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "experiment_state.json"


def load_state(path: Path = STATE_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_write(data: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def update_state(**changes) -> dict:
    state = load_state()
    state.update(changes)
    atomic_json_write(state)
    return state


def status_text(state: dict | None = None) -> str:
    state = load_state() if state is None else state
    return (
        f"Milestone: {state['milestone']} ({state['status']})\n"
        f"Active experiment: {state['active_experiment'] or 'none'}\n"
        f"Last completed step: {state['last_completed_step']}\n"
        f"Latest checkpoint: {state['latest_checkpoint'] or 'none'}\n"
        f"Next action: {state['next_action']}"
    )
