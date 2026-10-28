"""Integration check for the notebook's new-approach and resume controls."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runner import ExperimentConfig, run_session
from lab import compare_artifacts, load_artifact


def main():
    with tempfile.TemporaryDirectory(prefix="kan-lab-controls-") as name:
        root = Path(name)
        directory = root / "explorations"
        config = ExperimentConfig(run_id="mlp_tanh_trial", model="mlp", widths=[1, 4, 1],
                                  activation="tanh", seed=31, batch_size=32, validation_size=64,
                                  steps=4, checkpoint_every=2, validate_every=2)
        assert run_session(config, directory, max_steps_per_session=2)["status"] == "interrupted"
        assert run_session(config, directory, resume=True)["status"] == "completed"
        alternative = ExperimentConfig(run_id="kan_w1_trial", widths=[1, 2, 1], grid_intervals=4,
                                       loss="w1", seed=32, batch_size=32, validation_size=64,
                                       steps=4, checkpoint_every=2, validate_every=2)
        run_session(alternative, directory)
        rows, z, target, outputs = compare_artifacts(
            [config.run_id, alternative.run_id], sample_size=128, seed=801, root=root)
        assert len(rows) == 2 and all(value.shape == (128,) for value in outputs.values())
        assert load_artifact(config.run_id, root).provenance["status"] == "completed"
        try:
            compare_artifacts([config.run_id], sample_size=16, target_std=0.3, root=root)
            raise AssertionError("a mismatched target comparison was accepted")
        except ValueError:
            pass
    print("Notebook exploration, MLP/tanh, KAN/W1, explicit resume, and fair-comparison controls passed")


if __name__ == "__main__":
    main()
