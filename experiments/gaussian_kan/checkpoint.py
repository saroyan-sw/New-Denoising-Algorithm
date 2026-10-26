"""Atomic PyTorch checkpoint IO. Only load checkpoints created by this study."""

import os
import tempfile
from pathlib import Path

import torch


def save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        with open(name, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=map_location, weights_only=False)
