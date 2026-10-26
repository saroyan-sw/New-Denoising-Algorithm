"""Plot the initial and pilot KAN distributions and maps for inspection."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

from checkpoint import load_checkpoint
from models import make_kan
from state import ROOT


def draw(checkpoints: list[Path], output: Path) -> None:
    rng = np.random.default_rng(20261027)
    z_samples = rng.uniform(1e-7, 1 - 1e-7, size=20_000).astype("float32")
    z_grid = np.linspace(1e-4, 1 - 1e-4, 1000, dtype="float32")
    fig, axes = plt.subplots(len(checkpoints), 2, figsize=(11, 4 * len(checkpoints)), constrained_layout=True)
    if len(checkpoints) == 1:
        axes = np.expand_dims(axes, 0)
    for row, path in enumerate(checkpoints):
        payload = load_checkpoint(path)
        cfg = payload["config"]
        model = make_kan(cfg["widths"], cfg["grid_intervals"], cfg["spline_order"])
        model.load_state_dict(payload["model_state"])
        model.eval()
        with torch.no_grad():
            generated = model(torch.from_numpy(z_samples[:, None])).flatten().numpy()
            mapped = model(torch.from_numpy(z_grid[:, None])).flatten().numpy()
        target = rng.normal(0, cfg["target_std"], size=z_samples.size)
        axes[row, 0].hist(target, bins=90, range=(-0.8, 0.8), density=True, alpha=0.45, label="Target Gaussian")
        axes[row, 0].hist(generated, bins=90, range=(-0.8, 0.8), density=True, alpha=0.55, label="KAN output")
        axes[row, 0].set(xlabel="Value", ylabel="Density", title=f"Output law, step {payload['step']}")
        axes[row, 0].legend()
        axes[row, 1].plot(z_grid, cfg["target_std"] * norm.ppf(z_grid), label="Increasing inverse CDF", lw=2)
        axes[row, 1].plot(z_grid, mapped, label="Learned KAN", lw=1.5)
        axes[row, 1].set(xlabel="Uniform input z", ylabel="Output", title=f"Map, step {payload['step']}", ylim=(-0.8, 0.8))
        axes[row, 1].legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=ROOT / "checkpoints" / "initial.pt")
    parser.add_argument("--current", type=Path, default=ROOT / "checkpoints" / "latest.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "figures" / "pilot_initial_and_step30.png")
    args = parser.parse_args()
    draw([args.initial, args.current], args.output)


if __name__ == "__main__":
    main()
