"""Replay pilot steps 21–30 from step 20 and compare exact saved weights."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from checkpoint import load_checkpoint
from models import make_kan
from train import restore_rng, uniform_open, wasserstein2_squared


def main() -> None:
    start = load_checkpoint(ROOT / "checkpoints" / "best_step_000020.pt")
    end = load_checkpoint(ROOT / "checkpoints" / "best_step_000030.pt")
    assert start["step"] == 20 and end["step"] == 30, (start["step"], end["step"])
    cfg = start["config"]
    model = make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    model.load_state_dict(start["model_state"])
    optimizer.load_state_dict(start["optimizer_state"])
    restore_rng(start["rng_state"])
    for _step in range(21, 31):
        latent = uniform_open((cfg["batch_size"], 1), "cpu")
        target = torch.randn((cfg["batch_size"], 1)) * cfg["target_std"]
        loss = wasserstein2_squared(model(latent), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, end["model_state"][key]), f"model mismatch: {key}"
    assert torch.equal(torch.get_rng_state(), end["rng_state"]["torch_cpu"]), "RNG state mismatch"
    assert len(end["optimizer_state"]["state"]) > 0
    print("Exact 10-step model and PyTorch RNG replay passed")


if __name__ == "__main__":
    main()
