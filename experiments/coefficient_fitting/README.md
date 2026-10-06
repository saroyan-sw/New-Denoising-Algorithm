# Low / mid / high coefficient fitting comparison

This experiment asks how to choose the three gains in the complementary-band estimator

\[
\hat{x}=c_L L+c_M M+c_H H,
\quad L=G_2(y),\quad M=G_1(y)-G_2(y),\quad H=y-G_1(y).
\]

Every case starts from a clean image, adds a reproducible Gaussian-noise realization
(`sigma = 25/255`), then fits or applies coefficients to the noisy observation. The
noise is left unclipped for the SURE objective. The restored image is clipped only
when computing display-domain PSNR, SSIM, and MSE.

## Methods exercised

The global fitting search compares grid search, random search, coordinate descent,
finite-difference gradient descent, Gaussian-process UCB search, and a compact
covariance-adapting evolution strategy. Each search is run against a clean-target MSE
objective (an oracle ceiling) and a Gaussian SURE objective. Fitted global weights are
learned on DIV2K validation crops, then evaluated on disjoint CBSD68 and Kodak24
images. Test-time per-image and patchwise fits are also reported for SURE and for
clean-target MSE as an oracle ceiling.

Raw image-level results, fitting-search comparisons, grouped summaries, and availability
of the remaining discussed objectives are saved under `results/`.

## Run

From the project root:

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_experiment.py
```

Defaults use 16 DIV2K validation calibration images and 8 images each from CBSD68 and
Kodak24, with fixed-size random crops and a recorded seed. This is a pilot; increase
`--train-images` and `--test-images` for the full datasets after reviewing the pilot.

BRISQUE/NIQE quality-only selection and the HITS filter ensemble need PyTorch/PyIQA.
Both discriminator checkpoints are copied into the project's `checkpoints/` folder.
`run_quality_methods.py` compares quality-only selection with a SURE-risk constraint;
`run_discriminator_methods.py` runs a deliberately short, per-image checkpoint-guided
pilot. The latter is a smoke test, not a tuned benchmark. Method status and limits are
recorded in `results/method_status.json` and `results/report.md`.

To rerun the archive-faithful discriminator experiment (Gaussian/Gabor/residual bands,
150 steps, clipped AWGN, and both mean-constraint variants):

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_archive_discriminator.py
```

Its default pilot fits four CBSD68 and four Kodak24 crops and writes to
`results/discriminator_archive_faithful/`.

To run the same noisy-input, archive-band coefficient fit with the public NIMA
technical-quality model (trained on TID2013), first install the `quality` extra
alongside PyTorch, then run:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[torch,quality]"
.venv\Scripts\python.exe experiments\coefficient_fitting\run_nima_technical.py
```

NIMA is a no-reference image-quality assessor, not a GAN discriminator. Its score
guides the fit alongside the same noisy-image fidelity and mean-brightness terms;
clean references are used only for PSNR/SSIM evaluation. Outputs and fitted weights
are saved under `results/nima_technical/`, including the restored pilot crops.

The 100-crop replication (50 CBSD68 and 50 Kodak24 crops) uses batched independent
per-crop coefficients and also writes clean/noisy/restored images plus a visualization:

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_nima_100_visual.py
```

Its recorded outputs are under `results/nima_technical_100/`.

To run the archive discriminator on those exact same 100 noisy crops and build the
large-view, side-by-side comparison:

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_discriminator_100_compare_visual.py
```

The discriminator outputs and paired NIMA/discriminator metrics are saved under
`results/discriminator_100/`. Add `--reuse-existing` to rebuild only the visualization.

The salt-and-pepper stress test uses the same 100-crop dataset structure, replaces 10%
of whole RGB pixels (5% black and 5% white), and compares both learned-guidance methods
with 3x3/5x5 median filters. It also records a deliberately mismatched Gaussian-SURE
diagnostic and a clean-reference three-band oracle ceiling:

```powershell
.venv\Scripts\python.exe experiments\coefficient_fitting\run_salt_pepper_100.py
```

Outputs are under `results/salt_pepper_100/`.

## Reading the comparison

The clean-target fits are diagnostic upper bounds and cannot be used as unsupervised
test-time methods. Global SURE tests whether a single validation-fitted vector transfers.
Per-image and patchwise SURE use only the noisy test observation, but their adaptive
selection and finite-sample SURE bias must be considered when interpreting results.
Coefficient search speed is not compared here; the experiment compares selected
coefficients and denoising quality.
