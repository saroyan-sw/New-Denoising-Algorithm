# Audited Image Denoising Baselines

Current experiment status and decisions: [docs/experiment_state_2026-10-02.md](docs/experiment_state_2026-10-02.md).

This repository reconstructs the useful parts of the earlier Self-FuseNet-inspired denoising project without carrying forward duplicated scripts, hard-coded paths, hidden package installation, or scientifically invalid comparisons.

The current code is a baseline, not a claim of state-of-the-art performance. It contains:

- deterministic Gaussian, bilateral, Sobel, Gabor, and overlap-add operators;
- a stable paper-inspired low/mid/high feature pipeline;
- a complementary three-band decomposition that exactly reconstructs its input at unit weights;
- full-reference evaluation metrics and reproducible synthetic degradations;
- a patch-wise selector that compares noise-specific classical denoisers with the archived pretrained clean discriminator;
- an optional PyTorch implementation for frozen-prior, per-image optimization;
- experiment and research-state documentation preserving the full history.

The most important conclusion of the audit is that no supplied artifact establishes a competitive denoising result. The two discriminator checkpoints are recoverable, but the result CSVs, logs, seeds, and saved Self-Fuse outputs are absent. Several scripts also contain comparison-breaking defects. See [the research-state report](docs/research_state_report.md) and [experiment ledger](docs/experiment_ledger.md).

The next-stage task and dataset decision is documented in [the benchmark and naturalness-prior strategy](docs/benchmark_strategy.md). Its machine-readable first protocol is [configs/benchmark_awgn_sigma25.json](configs/benchmark_awgn_sigma25.json).

The first implementation of the proposed book-reader/HITS analogy is documented in [the spectral filter-ensemble experiment](docs/spectral_ensemble_experiment.md). The required dataset layout is in [data/README.md](data/README.md). It learns global coefficients on DIV2K validation and evaluates the frozen mixture on CBSD68 and Kodak24.

## Quick start

On Windows, create the project environment and install the optional PyTorch and
no-reference metric dependencies with:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[torch]"
```

The checked-in project setup includes the two archived discriminator checkpoints in
`checkpoints/`. The environment used for the verified pilot is Python 3.14, PyTorch
2.14.0 (CPU), torchvision 0.29.0, and PyIQA 0.1.16. On Windows, PyTorch also needs a
current Microsoft Visual C++ x64 runtime. PyIQA model files are cached under the
project's ignored `.cache/torch/` directory.

Run the project's tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To rerun the low/mid/high coefficient pilot, including BRISQUE/NIQE and the short
checkpoint-guided test:

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_experiment.py
.venv\Scripts\python.exe experiments\coefficient_fitting\run_quality_methods.py
.venv\Scripts\python.exe experiments\coefficient_fitting\run_discriminator_methods.py
.venv\Scripts\python.exe experiments\coefficient_fitting\run_archive_discriminator.py
```

The experiment README documents crop sizes, image counts, and limits. Results are
written to `experiments/coefficient_fitting/results/`.

Run the matched 100-crop Simoncelli-style steerable-pyramid pilot (requires the
optional PyTorch and NIMA dependencies):

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_simoncelli_steerable_100.py
```

The pilot compares band gains alone with gains plus soft thresholds, using the
same crops, noise, and NIMA objective as the earlier 100-crop runs. See the
[results and interpretation](docs/simoncelli_pyramid_pilot_2026-10-02.md) and
open `experiments/coefficient_fitting/results/simoncelli_steerable_100_matched150/comparison.html`
for full-resolution comparisons. The [matched run report](docs/simoncelli_pyramid_matched_2026-10-02.md)
compares its 150-step results with the previous methods.

Run the dependency-light tests with the bundled or any normal Python environment containing NumPy and Pillow:

```powershell
python -m unittest discover -s tests -v
```

Run the deterministic baseline:

```powershell
python scripts/run_baseline.py input.png --output outputs/baseline.png
```

Run the patch-wise denoiser selector with the archived clean-domain discriminator:

```powershell
python scripts/run_patch_selection.py input.png --output outputs/patch_selected.png --noise-type awgn --selection-map outputs/patch_selection.png
```

The selected noise type controls the candidate bank (`awgn` or `salt_pepper`). Each
candidate processes the full image; overlapping patches are scored by the mean logit
from both scales of `checkpoints/netD_Clean_epoch_100.pth`. Higher clean-domain logits
win, and winning patches are blended with Hann weights. The selection map and a JSON
sidecar are saved alongside the output. This discriminator is a learned naturalness
prior rather than a calibrated IQA metric, so the output should be evaluated against
clean references on a held-out benchmark; a high discriminator score alone does not
establish better denoising.

Run a reproducible synthetic benchmark on a directory of clean images:

```powershell
python scripts/run_benchmark.py path/to/clean_images --output outputs/benchmark.csv --seed 7
```

Fit and evaluate the no-reference spectral ensemble (requires the optional `torch` dependencies):

```powershell
python scripts/run_spectral_ensemble.py fit data/DIV2K_valid --output-dir outputs/spectral_sigma25 --sigma 25
python scripts/run_spectral_ensemble.py evaluate data/CBSD68 --weights outputs/spectral_sigma25/weights.json --output-dir outputs/cbsd68_sigma25
```

Run the corrected frozen-prior optimizer globally or with dense overlap-add patches:

```powershell
python scripts/run_prior_optimization.py input.png path/to/netD_Clean_epoch_100.pth --output outputs/prior.png --mode patch
```

The optional differentiable baseline requires the `torch` extra. It deliberately does not auto-install packages. A trained discriminator must be supplied explicitly; the archived checkpoints are documented in [source inventory](docs/source_inventory.md).

## Dataset size plots

Generate per-image dimension and file-size statistics, then save Matplotlib PNG and SVG charts:

```powershell
python -m pip install -e ".[viz]"
python scripts/report_image_stats.py
python scripts/plot_image_stats.py
```

The charts are saved to `outputs/image_stats/dataset_sizes.png` and `.svg`.

## What is intentionally not included

- the original rapid-experiment scripts;
- claimed results that cannot be recovered;
- the broken coefficient normalization used by several later scripts;
- the mislabeled one-scale derivative-of-Gaussian bank called a Simoncelli steerable pyramid;
- hard-coded local dataset and checkpoint paths;
- automatic `pip install` calls at import time.

These decisions preserve the experimental history in documentation while keeping executable code small, testable, and honest about what has and has not been verified.
