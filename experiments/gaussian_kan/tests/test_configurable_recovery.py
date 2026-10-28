"""Exact full-versus-resumed KAN training and overwrite protection."""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from checkpoint import load_checkpoint
from experiment_runner import ExperimentConfig, distribution_loss, run_session


def main():
    x = torch.tensor([0.3, -0.2, 0.1], requires_grad=True)
    y = torch.tensor([-0.5, 0.4, 0.2])
    loss = distribution_loss(x, y)
    assert torch.equal(loss, distribution_loss(x[[2, 0, 1]], y[[1, 2, 0]]))
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    config = ExperimentConfig(run_id="full", widths=[1, 2, 1], grid_intervals=4,
                              batch_size=32, validation_size=64, steps=6,
                              validate_every=2, checkpoint_every=2, seed=75)
    with tempfile.TemporaryDirectory(prefix="kan-replay-") as name:
        directory = Path(name)
        assert run_session(config, directory)["status"] == "completed"
        split = replace(config, run_id="split")
        assert run_session(split, directory, max_steps_per_session=2)["status"] == "interrupted"
        assert run_session(split, directory, resume=True)["status"] == "completed"
        full = load_checkpoint(directory / "full/checkpoints/latest.pt")
        resumed = load_checkpoint(directory / "split/checkpoints/latest.pt")
        assert all(torch.equal(value, resumed["model_state"][key]) for key, value in full["model_state"].items())
        assert torch.equal(full["rng_state"]["torch_cpu"], resumed["rng_state"]["torch_cpu"])
        for key, entry in full["optimizer_state"]["state"].items():
            assert all(torch.equal(value, resumed["optimizer_state"]["state"][key][item]) for item, value in entry.items())
        try:
            run_session(split, directory)
            raise AssertionError("existing run was overwritten")
        except FileExistsError:
            pass
        try:
            run_session(replace(split, learning_rate=0.002), directory, resume=True)
            raise AssertionError("changed config was accepted on resume")
        except ValueError:
            pass
    print("Exact KAN model/optimizer/RNG replay, loss invariance, and overwrite protection passed")


if __name__ == "__main__":
    main()
