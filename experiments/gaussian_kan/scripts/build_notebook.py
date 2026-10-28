"""Author the reproducible step-by-step Gaussian KAN research notebook."""

from pathlib import Path
from textwrap import dedent

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nb.v4.new_markdown_cell(dedent(source).strip()))


def code(source):
    cells.append(nb.v4.new_code_cell(dedent(source).strip()))


md("""
# Gaussian KAN laboratory

**Completed research and a workspace for your own experiments — 29 October 2026.**

Can a spline KAN transform independent uniform samples into an approximately Gaussian output law using unpaired target samples?

The measured answer is **yes, approximately**. The KAN learned a decreasing Gaussian quantile-like map. It beat the tested MLP, but its tails remain imperfect and a Gaussian parameter fit is much cheaper and more accurate when the Gaussian assumption is available.

Run the cells in order. Default execution checks the saved evidence and performs only a tiny recovery demonstration. Full study runs are loaded from saved checkpoints. New training is controlled by an explicit switch near the end.

Keep this notebook beside the study's Python modules and results. The portable bundle also includes checkpoints. A Git-only checkout can display the saved tables and figures, while weight-dependent cells need the checkpoint bundle or a deliberate rerun.
""")

md("""
## The route through the experiment

1. Check the environment and configuration.
2. Generate independent latent and target sample sets.
3. Verify the sorted-sample loss, KAN forward pass, gradients, and updates.
4. Inspect checkpoints and demonstrate exact recovery.
5. Review training progress, baseline results, quantiles, and tails.
6. Compare seeds and spline capacity.
7. Choose an approach, train a new named run, resume it, and compare its output.

Distribution matching does not uniquely identify a physical stochastic mechanism. Increasing and decreasing quantile maps can generate the same Gaussian law.
""")

code(r"""
from pathlib import Path
import os, sys, json, csv, math, hashlib, inspect, tempfile, subprocess
from dataclasses import asdict, replace
import importlib.metadata as package_metadata

here = Path.cwd().resolve()
candidates = [here, here / "experiments/gaussian_kan",
              here / "New-Denoising-Algorithm/experiments/gaussian_kan", here.parent]
ROOT = next((p for p in candidates if (p / "config/experiment.yaml").exists()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from the gaussian_kan study folder or its repository root.")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.special import ndtri
from IPython.display import display, Image, Code, Markdown
get_ipython().run_line_magic("matplotlib", "inline")

from models import make_kan, make_mlp, parameter_count
from checkpoint import load_checkpoint
from train import uniform_open, wasserstein2_squared
from experiment_runner import ExperimentConfig, run_session, distribution_loss
from lab import load_artifact, compare_artifacts

METRICS = ROOT / "results/metrics"
FIGURES = ROOT / "results/figures"
HAS_WEIGHTS = (ROOT / "checkpoints/latest.pt").exists()
print("Study folder:", ROOT)
print("Saved model weights available:", HAS_WEIGHTS)
""")

md("""
## 1. Environment and fixed protocol

The original study used CPU because CUDA was unavailable. The KAN library is a genuine B-spline implementation, pinned to a specific source commit. Target parameters appear in data sampling and evaluation; the generator receives only latent values.
""")

code("""
versions = {name: package_metadata.version(name) for name in
            ["torch", "numpy", "scipy", "matplotlib", "pandas", "efficient-kan", "nbformat", "ipykernel"]}
display(pd.Series(versions, name="Installed version").to_frame())
print("Python:", sys.version.split()[0], "| CUDA:", torch.cuda.is_available(),
      "| PyTorch CPU threads:", torch.get_num_threads())
study_config = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
display(pd.Series(study_config, name="Primary experiment").to_frame())
""")

md("""
## 2. Independent, unpaired samples

The main study samples Z uniformly over the open unit interval and N from a zero-mean Gaussian with standard deviation 0.2. They are independent sample sets; there is no clean/noisy pair and no known target output for an individual latent value. Exact numerical endpoints are excluded.
""")

code("""
latent_rng = torch.Generator().manual_seed(101)
target_rng = torch.Generator().manual_seed(202)
z_demo = uniform_open((4096, 1), "cpu", latent_rng)
n_demo = torch.randn((4096, 1), generator=target_rng) * 0.2
assert ((z_demo > 0) & (z_demo < 1)).all()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].hist(z_demo.numpy().ravel(), bins=35, density=True)
axes[0].set(title="Independent latent sample", xlabel="z", ylabel="Density")
axes[1].hist(n_demo.numpy().ravel(), bins=45, density=True)
axes[1].set(title="Independent target sample", xlabel="n", ylabel="Density")
plt.tight_layout(); plt.show()
""")

md("""
## 3. Check the distribution loss

Sort generated and target samples independently, then average squared differences. This is empirical squared one-dimensional Wasserstein-2 distance for equal-sized samples. Permuting either set must leave it unchanged. Ordinary MSE between unrelated sample indices is not the objective.
""")

code("""
generated_toy = torch.tensor([0.3, -0.2, 0.1, 0.05], requires_grad=True)
target_toy = torch.tensor([-0.4, 0.2, 0.1, 0.02])
loss_before = wasserstein2_squared(generated_toy, target_toy)
loss_permuted = wasserstein2_squared(generated_toy[[2, 0, 3, 1]], target_toy[[1, 3, 0, 2]])
assert torch.equal(loss_before, loss_permuted)
loss_before.backward()
assert torch.isfinite(generated_toy.grad).all() and generated_toy.grad.abs().sum() > 0
display(pd.DataFrame({
    "Quantity": ["Sorted squared W2", "After independent permutations", "Unpaired index-wise MSE"],
    "Value": [loss_before.item(), loss_permuted.item(),
              (generated_toy.detach() - target_toy).square().mean().item()]
}))
display(Code(inspect.getsource(distribution_loss), language="python"))
""")

md("""
## 4. Inspect the spline KAN and one update

The primary network has widths [1, 8, 8, 1], 32 grid intervals, cubic B-splines, and 2,960 trainable parameters. Each edge has a learned univariate function. The library combines a base SiLU term with a B-spline expansion. This cell creates a temporary model; the saved study weights are not used for the update.
""")

code("""
torch.manual_seed(314)
temporary_kan = make_kan([1, 8, 8, 1], 32, 3)
optimizer = torch.optim.Adam(temporary_kan.parameters(), lr=0.001)
before = [p.detach().clone() for p in temporary_kan.parameters()]
latent = uniform_open((128, 1), "cpu")
target = torch.randn((128, 1)) * 0.2
output = temporary_kan(latent)
loss = wasserstein2_squared(output, target)
optimizer.zero_grad(set_to_none=True); loss.backward()
assert output.shape == (128, 1)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in temporary_kan.parameters())
optimizer.step()
change = max(float((a - p.detach()).abs().max()) for a, p in zip(before, temporary_kan.parameters()))
assert change > 0
display(pd.Series({"Parameters": parameter_count(temporary_kan), "Loss before update": loss.item(),
                   "Largest parameter change": change}, name="Forward/gradient/update check").to_frame())
display(Code(inspect.getsource(make_kan), language="python"))
""")

md("""
## 5. Checkpoints and faithful recovery

Each checkpoint includes weights, Adam state, step/configuration, Python/NumPy/PyTorch random states, validation measurements, and a reference to the best validation checkpoint. Atomic writes and periodic checkpoints provide recovery even when a shutdown hook cannot run.

The following inventory verifies selected checkpoint hashes from the recorded manifest. The next cell runs a tiny network continuously and in two sessions, then compares weights, optimizer tensors, and random state exactly.
""")

code("""
manifest = json.loads((ROOT / "CHECKPOINT_MANIFEST.json").read_text())
inventory = pd.DataFrame(manifest["checkpoints"])
print("Recorded checkpoints:", len(inventory))
selected_paths = ["checkpoints/initial.pt", "checkpoints/step_000100.pt",
                  "checkpoints/best_step_000400.pt", "checkpoints/latest.pt"]
selected = inventory[inventory["path"].isin(selected_paths)].copy()
selected["available"] = selected["path"].map(lambda p: (ROOT / p).is_file())
if HAS_WEIGHTS:
    for row in selected.to_dict("records"):
        assert hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    checkpoint = load_checkpoint(ROOT / "checkpoints/latest.pt")
    assert checkpoint["step"] == 1000 and len(checkpoint["optimizer_state"]["state"]) > 0
    print("Final primary checkpoint random states:", sorted(checkpoint["rng_state"]))
display(selected[["path", "step", "bytes", "available", "sha256"]])
""")

code("""
(ROOT / "explorations").mkdir(exist_ok=True)
demo_config = ExperimentConfig(run_id="continuous", widths=[1, 2, 1], grid_intervals=4,
                               seed=75, batch_size=32, validation_size=64,
                               steps=6, validate_every=3, checkpoint_every=3)
with tempfile.TemporaryDirectory(prefix="notebook_recovery_", dir=ROOT / "explorations") as folder:
    folder = Path(folder)
    run_session(demo_config, folder)
    split_config = replace(demo_config, run_id="split")
    first_session = run_session(split_config, folder, max_steps_per_session=3)
    second_session = run_session(split_config, folder, resume=True)
    a = load_checkpoint(folder / "continuous/checkpoints/latest.pt")
    b = load_checkpoint(folder / "split/checkpoints/latest.pt")
    assert all(torch.equal(value, b["model_state"][key]) for key, value in a["model_state"].items())
    assert torch.equal(a["rng_state"]["torch_cpu"], b["rng_state"]["torch_cpu"])
    for key, entry in a["optimizer_state"]["state"].items():
        assert all(torch.equal(value, b["optimizer_state"]["state"][key][item]) for item, value in entry.items())
print("Exact weights, optimizer state, and PyTorch RNG replay: PASSED")
print("First session:", first_session["status"], "| resumed session:", second_session["status"])
""")

md("""
## 6. Follow the primary training run

The saved primary trajectory started with a very narrow output distribution, improved through 1,000 steps, and selected step 400 by fixed-validation W2. Training and validation fluctuate because fresh unpaired mini-batches have finite-sample variation. Best and final checkpoints are both preserved.
""")

code("""
history = pd.read_csv(METRICS / "kan_primary_seed_1729.csv")
display(history[history["step"].isin([0, 20, 30, 100, 400, 1000])])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(history["step"], np.sqrt(history["train_w2_squared"]), "--", label="Train batch W2")
axes[0].plot(history["step"], np.sqrt(history["val_w2_squared"]), label="Fixed validation W2")
axes[0].set(yscale="log", xlabel="Step", ylabel="W2", title="Primary KAN progress")
axes[0].legend()
axes[1].plot(history["step"], history["output_std"], label="Generated standard deviation")
axes[1].axhline(0.2, color="black", linestyle="--", label="Target 0.2")
axes[1].set(xlabel="Step", ylabel="Standard deviation")
axes[1].legend()
plt.tight_layout(); plt.show()
display(Image(filename=str(FIGURES / "distribution_snapshots.png"), width=1100))
""")

md("""
## 7. Compare the completed baselines

These are the **first independent test** results: 100,000 uniform inputs and an independent 100,000-sample target set, unused for checkpoint selection. The MLP has 2,913 parameters, approximately matching the KAN. The Gaussian parameter fit and oracle explicitly use Gaussian knowledge. The empirical affine fit uses sorted unpaired training samples without assuming a Gaussian family.

The validation-selected checkpoint and final checkpoint are reported separately. A test result does not change the preselected checkpoint.
""")

code("""
test_results = pd.read_csv(METRICS / "independent_test_metrics.csv")
display(test_results[["method", "test_w2", "mean_error", "std_error", "parameters",
                      "selected_step", "training_loop_seconds", "uses_gaussian_family"]].round(6))
floor = json.loads((METRICS / "sampling_floor.json").read_text())
print(f"Gaussian-versus-Gaussian W2 reference: {floor['mean_w2']:.6f} ± {floor['sd_w2']:.6f} (SD, 10 pairs)")
display(Image(filename=str(FIGURES / "baseline_comparison.png"), width=1000))
""")

md("""
## 8. Inspect any saved generators yourself

Edit the method list, sample size, and seed below. This computes a fresh comparison without training or altering saved weights. The sample variation can change small differences between methods. Use the published tables for the recorded study results.

Available aliases: kan_best, kan_final, mlp_best, affine_best, affine_empirical, gaussian_fit, oracle. You can also use any completed robustness run ID or your own saved exploration run ID.
""")

code("""
METHODS_TO_INSPECT = ["kan_best", "mlp_best", "gaussian_fit"]
INSPECTION_SIZE = 20000
INSPECTION_SEED = 7101

if HAS_WEIGHTS:
    comparison, latent_view, target_view, output_view = compare_artifacts(
        METHODS_TO_INSPECT, sample_size=INSPECTION_SIZE, seed=INSPECTION_SEED)
    display(pd.DataFrame(comparison).round(6))
    grid = np.linspace(1e-4, 1 - 1e-4, 1500, dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    edges = np.linspace(-0.8, 0.8, 85)
    axes[0].hist(target_view, bins=edges, density=True, histtype="step", lw=2, label="Independent target")
    for name in METHODS_TO_INSPECT:
        axes[0].hist(output_view[name], bins=edges, density=True, histtype="step", label=name)
        axes[1].plot(grid, load_artifact(name).predict(grid), label=name)
    axes[1].plot(grid, .2 * ndtri(grid), "k--", label="Increasing oracle")
    axes[1].plot(grid, -.2 * ndtri(grid), "k:", label="Decreasing oracle")
    axes[0].set(xlabel="Output", ylabel="Density", title="Fresh output comparison")
    axes[1].set(xlabel="Uniform input z", ylabel="Output", title="Actual maps", ylim=(-.8, .8))
    for ax in axes: ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
else:
    print("Load the checkpoint bundle to run live saved-model comparisons. Published evidence remains visible below.")
""")

md("""
## 9. Quantiles, monotonicity, and tails

A similar histogram is insufficient. The selected KAN map was strictly decreasing on 10,000 interior points and had RMSE 0.00642 against the decreasing quantile reference. The inverse CDF is unbounded at z=0 and z=1, whereas these finite models flatten. Extreme-tail errors remain visible in quantiles and exceedance probabilities.
""")

code("""
diagnostics = pd.DataFrame(json.loads((METRICS / "function_diagnostics.json").read_text())).T
display(diagnostics[["orientation", "interior_rmse_increasing", "interior_rmse_decreasing",
                     "fraction_increasing_differences", "fraction_decreasing_differences",
                     "interior_max_absolute_error_chosen"]])
quantiles = pd.read_csv(METRICS / "quantile_errors.csv")
display(quantiles[quantiles["probability"].isin([.001, .01, .99, .999])]
        .pivot(index="method", columns="probability", values="signed_error").round(6))
tails = pd.read_csv(METRICS / "tail_fractions.csv")
display(tails[tails["absolute_threshold"] == .6].round(6))
display(Image(filename=str(FIGURES / "independent_qq.png"), width=650))
display(Image(filename=str(FIGURES / "tail_quantile_errors.png"), width=850))
""")

md("""
## 10. Stability and how much capacity was needed

Two additional primary seeds, a smaller [1,8,1] KAN, and a 64-grid primary KAN were run under the fixed training protocol. The table uses a fresh confirmation pair. Primary three-seed W2 averaged 0.00584 (SD 0.00051). The 592-parameter KAN was the smallest tested and retained similar central-distribution accuracy. It does not establish a minimum. Finer resolution improved some tail errors, but optimization variation was comparable to the overall accuracy gain.

Timing across different sessions is indicative: the first primary run used another runner/session. Compare same-session timings or repeat controlled benchmarks before attributing runtime differences to architecture.
""")

code("""
robustness = pd.read_csv(METRICS / "robustness_comparison.csv")
display(robustness[["run_id", "parameters", "selected_step", "confirmation_w2",
                    "final_confirmation_w2", "interior_map_rmse", "q001_error",
                    "q999_error", "training_loop_seconds"]].round(6))
robust_summary = json.loads((METRICS / "robustness_summary.json").read_text())
print("Three primary seeds:", robust_summary["primary_three_seed_mean_w2"],
      "±", robust_summary["primary_three_seed_sd_w2"], "(sample SD)")
display(Image(filename=str(FIGURES / "robustness_comparison.png"), width=1000))
display(Image(filename=str(FIGURES / "capacity_maps.png"), width=1000))
""")

md("""
## 11. Try another approach

Choose a recipe and edit the configuration. Main-study results always use squared W2 and Adam. The W1 recipe is an optional different objective, and the MLP recipes allow an activation comparison. None of these optional recipes is reported as a completed research result unless you actually run it.

Use a unique run name for a new configuration. An existing run cannot be silently overwritten. To continue the same configuration, explicitly set RESUME_EXPERIMENT=True. Validation always measures squared W2, including when training with W1.
""")

code("""
APPROACH_RECIPES = {
    "small_kan": {"model": "kan", "widths": [1, 8, 1], "grid_intervals": 32},
    "primary_kan": {"model": "kan", "widths": [1, 8, 8, 1], "grid_intervals": 32},
    "finer_kan": {"model": "kan", "widths": [1, 8, 8, 1], "grid_intervals": 64},
    "mlp_silu": {"model": "mlp", "widths": [1, 52, 52, 1], "activation": "silu"},
    "mlp_tanh": {"model": "mlp", "widths": [1, 52, 52, 1], "activation": "tanh"},
    "affine": {"model": "affine", "widths": [1, 1]},
    "small_kan_w1": {"model": "kan", "widths": [1, 8, 1], "grid_intervals": 32, "loss": "w1"},
}
APPROACH = "small_kan"             # change to any recipe above
RUN_NAME = "my_small_kan_001"      # use a NEW name for changed configurations
EXPERIMENT = ExperimentConfig(
    run_id=RUN_NAME, **APPROACH_RECIPES[APPROACH],
    seed=777, steps=1000, batch_size=2048, learning_rate=0.001,
)
EXPERIMENT.check()
display(pd.Series(asdict(EXPERIMENT), name="Your editable experiment").to_frame())
""")

code("""
RUN_NEW_EXPERIMENT = False
RESUME_EXPERIMENT = False
MAX_SESSION_SECONDS = 60
MAX_SESSION_STEPS = 200

if RUN_NEW_EXPERIMENT:
    result = run_session(
        EXPERIMENT, ROOT / "explorations",
        resume=RESUME_EXPERIMENT,
        max_runtime_seconds=MAX_SESSION_SECONDS,
        max_steps_per_session=MAX_SESSION_STEPS,
    )
    display(pd.Series(result, name="Session result").to_frame())
else:
    print("New training is disabled. Set RUN_NEW_EXPERIMENT=True to execute the configuration above.")
    print("To continue it later, keep the same configuration and set RESUME_EXPERIMENT=True.")
""")

md("""
After your run has a saved checkpoint, add its run name to the comparison below. Interrupted runs can be inspected too; their status is displayed. Use the same comparison sample size and seeds across candidates. Keep a further untouched test sample if you use these comparisons to select an approach.
""")

code("""
USER_COMPARISON = ["kan_best", "kan_small32_seed_1729", RUN_NAME]
available = []
for name in USER_COMPARISON:
    try:
        artifact = load_artifact(name)
        available.append(name)
    except FileNotFoundError:
        print("Not run / weights unavailable:", name)
if available:
    rows, z_user, target_user, generated_user = compare_artifacts(
        available, sample_size=20000, seed=8801)
    display(pd.DataFrame(rows).round(6))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    edges = np.linspace(-.8, .8, 80)
    ax.hist(target_user, bins=edges, density=True, histtype="step", lw=2, label="Independent target")
    for name in available:
        ax.hist(generated_user[name], bins=edges, density=True, histtype="step", label=name)
    ax.set(xlabel="Output", ylabel="Density", title="Your controlled comparison")
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("""
## 12. Verify the saved study artifacts

This runs the recorded checkpoint, independent-evaluation, and robustness checks. It does not restart a completed 1,000-step experiment. The pilot recovery check deliberately replays ten steps in memory.
""")

code("""
RUN_ARTIFACT_CHECKS = True
if RUN_ARTIFACT_CHECKS and HAS_WEIGHTS:
    commands = [
        [sys.executable, str(ROOT / "scripts/build_checkpoint_manifest.py"), "--verify"],
        [sys.executable, str(ROOT / "tests/test_resume.py")],
        [sys.executable, str(ROOT / "tests/test_baseline_artifacts.py")],
        [sys.executable, str(ROOT / "tests/test_evaluation_artifacts.py")],
        [sys.executable, str(ROOT / "tests/test_robustness_artifacts.py")],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        print(completed.stdout.strip())
        if completed.returncode:
            print(completed.stderr)
            raise RuntimeError(f"Artifact check failed: {command[-1]}")
else:
    print("Artifact replay checks skipped; use the portable checkpoint bundle to enable them.")
""")

md("""
## Interpretation and next experiment

- The KAN learned an approximate Gaussian output law from unpaired samples and a monotone decreasing quantile-like transformation.
- Central approximation is good, but extreme tails and finite-batch optimization remain measurable limitations.
- A 592-parameter KAN was sufficient for similar central accuracy in the tested setting. The true minimum is unknown.
- This KAN beat the tested MLP under the recorded budget. Gaussian parameter estimation was cheaper and closer to the sampling reference.
- The three primary KAN seeds were similar; the architecture alternatives and MLP each have only one seed.
- Output-law agreement does not recover an unknown physical noise mechanism from noisy images.

Recommended next work: average validation across more draws, examine a learning-rate schedule and tail errors, then test unpaired non-Gaussian scalar distributions with predeclared comparisons. A later noisy-image study needs explicit assumptions separating image content and noise; this scalar result does not resolve that identification problem.

Read FINAL_RESEARCH_REPORT.md for the study conclusions, RESUME_INSTRUCTIONS.md for continuation, and CHECKPOINT_MANIFEST.json for artifact hashes.
""")

notebook = nb.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python (Gaussian KAN environment)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "title": "Gaussian KAN laboratory",
}
output = ROOT / "Gaussian_KAN_Lab.ipynb"
nb.write(notebook, output)
nb.validate(notebook)
print(f"Wrote {output.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
