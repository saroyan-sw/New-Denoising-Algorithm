# Experiment state — 2026-10-02

## Current research question

Can a constrained, interpretable low/mid/high-band image model be fitted from only a
noisy observation, using a learned quality signal or a statistical risk estimate, and
produce competitive denoising without paired training data?

## Implemented branches

1. **Archived discriminator guidance**
   - Bands: 9x9 Gaussian low, averaged 5x5 Gabor mid, residual high.
   - Three per-crop coefficients are converted to positive weights summing to one.
   - Objective: frozen CycleGAN clean discriminator + noisy-input L1 fidelity + mean
     brightness constraint.
   - Optimization: Adam, 150 steps.

2. **NIMA technical-quality guidance**
   - Same archived bands, coefficient rule, fidelity term, brightness constraint, and
     optimizer.
   - Frozen public NIMA MobileNet technical-quality model trained on TID2013 replaces
     the discriminator.

3. **Corrected complementary bands with SURE**
   - Bands: `L=G2(y)`, `M=G1(y)-G2(y)`, and `H=y-G1(y)`, so `L+M+H=y` exactly.
   - Coefficients are not normalized to sum to one; `(1,1,1)` preserves the input.
   - SURE selects coefficients using the noisy image and known AWGN standard deviation.
   - Multiple searches were exercised: grid, random, coordinate, finite-difference
     gradient, Gaussian-process UCB, and a compact covariance-adapting search.

4. **Classical controls**
   - Median 3x3 and 5x5 were evaluated on the paired salt-and-pepper test.
   - Earlier filter-bank experiments also include Gaussian, bilateral, median, and a
     nine-filter spectral mixture.

5. **Original Self-FuseNet paper**
   - Retained as a separate super-resolution reference, not labeled as a denoiser.
   - Its paper pipeline uses bilateral/Gabor/edge-enhancement fusion selected by
     BRISQUE/NIQE, followed by a per-image customized U-Net and scale-specific
     upsampling.

## Completed quantitative results

### AWGN, matched 100-crop learned-guidance test

Protocol: 50 CBSD68 and 50 Kodak24 crops, 128x128, clipped AWGN with
`sigma=25/255`, one recorded noise realization per crop.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Noisy input | 20.4595 dB | 0.38938 |
| NIMA-guided archived bands | **24.2778 dB** | **0.55940** |
| Discriminator-guided archived bands | 23.2517 dB | 0.52059 |

NIMA had higher PSNR than the discriminator on 92 of 100 crops.

### AWGN, corrected complementary-band pilot

This used a different crop size/subset and unclipped AWGN, so it is not a direct paired
comparison with the 100-crop table.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Noisy input | 20.4701 dB | 0.40933 |
| Global SURE + gradient search | 26.3630 dB | 0.69509 |
| Per-image SURE + gradient search | **27.1740 dB** | 0.74104 |
| Patchwise SURE + grid search | 27.1232 dB | **0.74566** |

The clean-target oracle and SURE selected nearly identical solutions in this pilot,
indicating that SURE is promising for known AWGN. A new matched 100-crop run is still
needed before comparing it numerically with NIMA or the discriminator.

### Salt-and-pepper noise, paired 100-crop test

Protocol: 50 CBSD68 and 50 Kodak24 crops, 128x128, 10% whole-pixel corruption:
5% black and 5% white.

| Method | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Noisy input | 15.1145 dB | 0.24952 |
| Median 3x3 | **29.3149 dB** | **0.83146** |
| Median 5x5 | 27.0803 dB | 0.69990 |
| NIMA-guided archived bands | 20.5522 dB | 0.40601 |
| Discriminator-guided archived bands | 20.3323 dB | 0.39836 |
| Gaussian-SURE proxy, invalid noise model | 23.0197 dB | 0.52602 |
| Clean-target complementary-band oracle | 24.1414 dB | 0.59924 |

Median 3x3 won 97 crops and median 5x5 won three. Standard SURE is not valid for
salt-and-pepper noise; its result is retained only as a deliberate mismatch diagnostic.

## Current interpretation

The main limitation is now considered **representational capacity**, not only coefficient
selection:

- Salt-and-pepper corruption is spatially sparse but spreads energy across all frequency
  bands. Three global linear gains cannot identify individual outlier pixels.
- Even the clean-target complementary-band oracle was 5.17 dB below median 3x3 on the
  salt-and-pepper test. Therefore, a better assessor or optimizer cannot rescue that
  representation for impulse noise.
- The archive coefficient normalization is structurally problematic. Because its bands
  sum to the input, exact identity needs coefficients `(1,1,1)`, but normalization forces
  their sum to one and excludes identity.
- The averaged Gabor response is not a proper complementary frequency partition.
- NIMA is a better selector than the archived discriminator, but both can reward
  smoothness while removing genuine texture.
- Corrected complementary Gaussian bands remain useful for known AWGN, where SURE has a
  statistically justified objective and performed well in the pilot.

## Decision at this checkpoint

- **Stop treating the archived normalized Gaussian/Gabor/residual model as the main
  candidate.** Preserve it as a documented negative baseline.
- **Continue the AWGN branch with identity-preserving complementary bands and SURE.**
- **Use impulse-aware nonlinear processing for salt-and-pepper noise**, beginning with
  median/adaptive-median filtering or a detected-pixel correction mask.
- Before more learned-quality optimization, test whether a richer representation has an
  adequate clean-target oracle ceiling: multiscale/oriented bands, nonlinear shrinkage,
  spatially varying coefficients, and an unchanged-pixel residual path.
- Run the next decisions as small pilots first; scale to 100 crops only after the pilot
  establishes value.

## Result locations

- `experiments/coefficient_fitting/results/nima_technical_100/`
- `experiments/coefficient_fitting/results/discriminator_100/`
- `experiments/coefficient_fitting/results/salt_pepper_100/`
- `experiments/coefficient_fitting/results/summary.csv`
- `experiments/coefficient_fitting/results/report.md`
- `experiments/coefficient_fitting/results/method_status.json`

The exact scripts, seeds, crops, coefficients, checkpoint hashes, per-image metrics, and
saved output images are retained in those folders and their corresponding experiment
scripts.
