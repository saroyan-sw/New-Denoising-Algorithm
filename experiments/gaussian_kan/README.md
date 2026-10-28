# Gaussian generator with a spline KAN

**Completed research — 29 October 2026.**

Open [Gaussian_KAN_Lab.ipynb](Gaussian_KAN_Lab.ipynb) for the executed step-by-step notebook, [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for its experiment controls, or [FINAL_RESEARCH_REPORT.md](FINAL_RESEARCH_REPORT.md) for conclusions. [ARTIFACTS.md](ARTIFACTS.md) describes the portable bundle.

The KAN learned an approximately Gaussian law from unpaired uniform/target samples, with a decreasing quantile-like map. It beat the tested parameter-matched MLP but retained extreme-tail error. A 592-parameter KAN had similar central accuracy; Gaussian parameter estimation was cheaper and more accurate when its family assumption was available.

## Open the notebook

From this study directory in the current project:

~~~powershell
& '..\..\.venv\Scripts\python.exe' -m jupyterlab Gaussian_KAN_Lab.ipynb
~~~

The default notebook reads saved results, checks loss/gradients/checkpoints, demonstrates exact recovery, and compares saved generators. Its 16 code cells executed without error. New training is disabled until you enable the experiment switch. Edit recipes for KAN widths/grids, MLP activations, affine generators, optimizer/loss, batch size, learning rate, and seed. New runs use separate names under explorations/ and explicit resume.

For a standalone copy, install requirements-notebook.txt and keep the notebook beside the code/results. The portable ZIP includes all 215 checkpoint files. A Git-only checkout retains saved notebook outputs and measurements; inspecting trained weights needs the bundle.

## Evidence and recovery

- [Baseline comparison](BASELINE_COMPARISON.md)
- [Independent scientific evaluation](SCIENTIFIC_EVALUATION.md)
- [Seed and capacity study](ROBUSTNESS_AND_CAPACITY.md)
- [Experiment plan](EXPERIMENT_PLAN.md), [current state](EXPERIMENT_STATE.md), and [resume instructions](RESUME_INSTRUCTIONS.md)
- CHECKPOINT_MANIFEST.json and environment_study.json record artifact hashes and the tested environment.

The primary KAN used the pure-PyTorch [efficient-kan dependency](https://github.com/shawcharles/efficient-kan/tree/0240ea138fe257d800c9bfa561025e1c44aedc71), widths [1,8,8,1], 32 intervals, and cubic B-splines. The Gaussian inverse CDF was evaluation only. Distribution matching does not identify an original stochastic mechanism.

Primary status: python train.py --status. A completed primary resume is a no-op. Future configurable runs use experiment_runner.py with a saved configuration, explicit --resume latest, and a session budget. Checkpoints and portable archives stay local; lightweight manifests, configuration, metrics, figures, and the executed notebook are version controlled.
