# Simoncelli style pyramid: matched 150-step comparison

## Matched conditions

The two pyramid variants were rerun with the low/mid/high NIMA experiment's 150 Adam updates, batch size 10, learning rate 0.01, NIMA technical checkpoint, and objective `-NIMA/10 + 10*L1(reconstruction,noisy) + 5*absolute mean-brightness difference`. During optimization the NIMA model receives the unclipped reconstruction, matching the earlier run. Final images are clipped for full-reference metric calculation.

The exact same 100 crops were reused: 50 CBSD68 and 50 Kodak24, 128×128, clipped AWGN sigma 25/255, seed 20261001. The script verifies crop identity and noisy-image PSNR against the saved NIMA run. Maximum pyramid identity reconstruction error was 4.77e-7.

## Results

| Method | Mean PSNR | Mean SSIM | Crops better than noisy PSNR |
|---|---:|---:|---:|
| Noisy input | 20.459 dB | 0.3894 | — |
| Pyramid, gains only | 20.462 dB | 0.3895 | 59/100 |
| Pyramid, gains + soft thresholds | 20.584 dB | 0.3970 | 92/100 |
| Previous low/mid/high NIMA fit | 24.278 dB | 0.5594 | 100/100 |
| Previous discriminator fit | 23.252 dB | 0.5206 | 100/100 |

The thresholded pyramid averages 0.124 dB above gain-only and 0.124 dB above the noisy input. It loses to the previous low/mid/high NIMA result on all 100 crops, with a mean paired difference of −3.694 dB. It loses to the discriminator result on 99 crops, with a mean difference of −2.668 dB.

The 40-step pilot performed better on this metric for the thresholded variant (21.553 dB). Extending optimization to 150 steps did not improve PSNR or SSIM. The visual-prior objective is not the same as PSNR, so this may reflect the prior/fidelity tradeoff or optimization trajectory; it is not evidence of a convergence benefit. The main result is that matching the step count does not close the large gap to the earlier low/mid/high fit.

## Artifacts

- Runner: `experiments/coefficient_fitting/run_simoncelli_steerable_100.py`
- Aggregate protocol and scores: `experiments/coefficient_fitting/results/simoncelli_steerable_100_matched150/summary.json`
- Per-crop scores and fitted parameters: `experiments/coefficient_fitting/results/simoncelli_steerable_100_matched150/metrics.csv`
- Restored images: `experiments/coefficient_fitting/results/simoncelli_steerable_100_matched150/images/`
- Full-resolution interactive comparison: `experiments/coefficient_fitting/results/simoncelli_steerable_100_matched150/comparison.html`
