# Simoncelli style steerable pyramid pilot

## Setup

`experiments/coefficient_fitting/run_simoncelli_steerable_100.py` implements a frequency domain, order 3 steerable tight frame with four log radial scales and four orientations. It uses fixed analysis and synthesis filters and fits a separate set of 16 band gains for each crop. The second variant also fits one complex soft threshold per band, preserving the coefficient phase. The coarsest residual is retained.

This is a custom Simoncelli style frequency pyramid, not a numerical reproduction of a particular toolbox implementation. It keeps all coefficient maps at full crop resolution and uses smooth complementary radial windows. The paired transform is checked before optimization; observed maximum identity reconstruction error was `4.77e-7`.

The quality prior and objective match the earlier NIMA run: `-NIMA/10 + 10*L1(restoration,noisy) + 5*absolute mean-brightness difference`. Gain bounds are `[0, 2]`. The run used 40 updates per crop and variant because the PyTorch environment is CPU only. This is shorter than the 150 updates used in the earlier low/mid/high baseline and makes this a pilot, not a convergence-matched comparison.

## Matched benchmark

All methods use the exact same 100 crops: 50 from CBSD68 and 50 from Kodak24, each 128×128, with clipped AWGN at sigma 25/255 and seed 20261001. The crop coordinates and noisy-image PSNR were checked against the saved NIMA results before the comparison was accepted.

| Method | Mean PSNR | Mean SSIM | PSNR wins over noisy |
|---|---:|---:|---:|
| Noisy input | 20.459 dB | 0.3894 | — |
| Steerable pyramid, gains only | 20.460 dB | 0.3894 | 59/100 |
| Steerable pyramid, gains + thresholds | 21.553 dB | 0.4306 | 100/100 |
| Earlier NIMA low/mid/high fit | 24.278 dB | 0.5594 | 100/100 |
| Earlier archive discriminator fit | 23.252 dB | 0.5206 | 100/100 |

The thresholded variant gains 1.094 dB over the noisy images on average, but trails the previous NIMA run by 2.725 dB and wins on only one crop. Gain only is effectively the identity operation: its average PSNR gain is 0.001 dB and its fitted mean gains remain close to one. The visual prior therefore used thresholds to denoise, while the fitted gains alone supplied almost no useful adjustment in this short run.

## Files

- `experiments/coefficient_fitting/results/simoncelli_steerable_100/summary.json`: protocol and aggregate results.
- `experiments/coefficient_fitting/results/simoncelli_steerable_100/metrics.csv`: per crop metrics and all fitted gains and thresholds.
- `experiments/coefficient_fitting/results/simoncelli_steerable_100/images/`: the 200 restored crops.
- `experiments/coefficient_fitting/results/simoncelli_steerable_100/comparison.html`: interactive full resolution comparison, including the prior NIMA and discriminator outputs.

## Interpretation

The representation reconstructs exactly and the thresholded filter improves over noisy input, so the test establishes that the pyramid is operational. It does not establish that the visual prior and parameterization are competitive. Optimization length differs from the prior run, so a stronger follow-up should first check convergence on a small fixed subset, then repeat the full matched set only if the scores improve materially.
