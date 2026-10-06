# Coefficient-fitting pilot results

Run date: 2026-10-01. Noise was added to every clean image before denoising: unclipped
AWGN with standard deviation `25/255`. Reported PSNR, SSIM, and MSE clip restored images
to `[0, 1]`. Fits use only noisy inputs except explicitly labeled clean-target oracle
diagnostics. This is a small pilot, not a benchmark claim.

## Three-band coefficient search

The base experiment fitted the complementary low/mid/high estimator on 16 random
128x128 DIV2K validation crops, then evaluated on eight random 192x192 crops each from
CBSD68 and Kodak24 (16 held-out crops total).

| Method | Mean PSNR (dB) | Mean SSIM | Mean MSE |
|---|---:|---:|---:|
| Noisy input | 20.4701 | 0.40933 | 0.0089882 |
| Global SURE + gradient search | 26.3630 | 0.69509 | 0.0027953 |
| Global oracle MSE + gradient search | 26.3631 | 0.69506 | 0.0027943 |
| Per-image SURE + gradient search | 27.1740 | 0.74104 | 0.0022485 |
| Patchwise SURE + grid search | 27.1232 | 0.74566 | 0.0023253 |

The global SURE and oracle fits selected nearly identical gains, around `(1, 1, 0.04)`.
Per-image SURE was close to the clean-target per-image oracle on this AWGN pilot. Grid,
random, coordinate, finite-difference gradient, Gaussian-process UCB, and covariance-
adapting evolution searches are compared in `search_comparison.csv`; budgets were not
matched, so the pilot does not establish a generally best optimizer.

## No-reference quality objectives

BRISQUE/NIQE were evaluated on a 125-point coefficient grid. Global quality-only and
risk-constrained selection both chose `(1.0, 1.5, 0.0)`. On the eight-image holdouts,
global selection averaged 25.7189 dB / 0.70645 SSIM for CBSD68 and 26.5036 dB / 0.63718
SSIM for Kodak24. A per-image quality-only objective collapsed toward poor outputs:
13.6215 dB on CBSD68 and 11.9215 dB on Kodak24. Adding the SURE risk ceiling prevented
that failure (25.6540 dB and 26.4519 dB respectively). This is direct evidence that
unconstrained no-reference metric optimization is vulnerable to metric exploitation.

Some extreme candidates caused invalid BRISQUE/NIQE values (56 and 58 failed values,
respectively). They were counted and penalized as worse than valid candidates; see
`nr_metric_failures.json`. The global constrained and unconstrained choices happened to
match in this pilot.

## HITS-style filter ensemble

A nine-filter global mixture was fitted using no-reference metric affinities on DIV2K
validation, then frozen and tested on separate eight-crop CBSD68 and Kodak24 subsets.
Mean held-out results for the spectral mixture were 26.7641 dB / 0.71846 SSIM on CBSD68
and 27.1709 dB / 0.68660 SSIM on Kodak24. Uniform filter fusion reached 26.6013 / 0.69702
and 26.9239 / 0.66374 respectively. The fitted mixture is a separate filter-bank method,
not the three-band coefficient model. Full per-filter data and weights are under
`hits/`, `hits_cbsd68/`, and `hits_kodak24/`.

## Checkpoint-guided coefficient pilots

The initial eight-step smoke test used a different Gaussian band basis and is preserved
under `discriminator/`; it should not be treated as a result for the archived algorithm.
The corrected run in `discriminator_archive_faithful/` reproduces the archive's 9x9
Gaussian low band, averaged 5x5 Gabor mid band, residual high band, absolute coefficient
normalization, Adam learning rate 0.01, L1 fidelity weight 10, and 150 optimizer steps.
Each of four CBSD68 and four Kodak24 128x128 crops received AWGN (`sigma=25/255`) and
was clipped to `[0,1]` before fitting, matching the archive's noisy-input range. Both
the no-mean and mean-constraint variants use the same noisy crops and frozen epoch-100
clean discriminator.

| Method | Mean PSNR (dB) | Mean SSIM | Mean MSE |
|---|---:|---:|---:|
| Noisy input | 20.4512 | 0.31283 | 0.0090221 |
| Archive bands, no mean constraint | 22.5196 | 0.41286 | 0.0056025 |
| Archive bands, with mean constraint | 23.0146 | 0.42765 | 0.0050083 |

On these eight pilot crops, the mean constraint added about 0.50 dB over the no-mean
variant. The result is now faithful to the archived per-image coefficient model and
150-step objective, but it is still a small crop pilot; it does not reproduce the
archive's 2x2 patch inference or establish general performance. The fitted low-band
weights were modest (mean about 0.031 without / 0.084 with the mean constraint), while
the Gabor-mid band received most weight. See its `per_image_results.csv` and
`fitted_coefficients.json` for image-level results and exact settings.

Both checkpoint-guided runs load the same exact archive epoch-100 checkpoint. The
architecture and strict checkpoint loading were verified, but the first smoke test's
different band model explains why its near-unit coefficients were not comparable.

## Public NIMA technical-quality model

The public NIMA technical head (MobileNet, trained on TID2013) was loaded from its
author-released weights via DeepInv. It is a no-reference image-quality assessor, not
a discriminator. The archive-faithful three-band model was fitted for 150 Adam steps
on the same eight clipped-AWGN crops used by the epoch-100 discriminator pilot. The
objective maximized the NIMA score while retaining the same L1 noisy-input fidelity
and mean-brightness terms. Clean references were used only for reported PSNR/SSIM/MSE.

| Method | Mean PSNR (dB) | Mean SSIM | Mean MSE |
|---|---:|---:|---:|
| Noisy input | 20.4512 | 0.31283 | 0.0090221 |
| CycleGAN discriminator, mean constraint | 23.0146 | 0.42765 | 0.0050083 |
| NIMA technical, mean constraint | 24.5810 | 0.48623 | 0.0035160 |

On this matched eight-crop pilot NIMA-guided fitting was better by about 1.57 dB over
the discriminator-guided run. This is encouraging but not yet evidence of a general
advantage: it is a small holdout, one noise draw per crop, and optimizing a learned
quality metric can exploit its preferences. Validate across all holdout images and
multiple seeds, and check perceptual artifacts before selecting it. Per-image metrics,
coefficients, scorer values, restored crops, and the checkpoint SHA-256 are recorded in
`nima_technical/`.

The follow-up 100-crop run used 50 reproducible 128x128 crops from CBSD68 and 50 from
Kodak24, each corrupted first with clipped AWGN at `sigma=25/255`. Mean PSNR improved
from 20.4595 to 24.2778 dB and mean SSIM from 0.38938 to 0.55940; PSNR improved on all
100 crops. Mean fitted low/mid/high weights were 0.2735 / 0.4736 / 0.2529. This larger
pilot is still crop-based and uses one noise realization per crop. Its exact crop
coordinates, metrics, saved clean/noisy/restored images, and metadata are under
`nima_technical_100/`.

The archive epoch-100 discriminator was subsequently run on the exact same 100 noisy
crops for a paired comparison. Its mean PSNR/SSIM were 23.2517 dB / 0.52059, compared
with 24.2778 dB / 0.55940 for NIMA guidance. NIMA had higher PSNR on 92 crops and the
discriminator on 8. The paired metrics and discriminator-restored images are under
`discriminator_100/`; both methods use the same bands, coefficient normalization,
150-step optimizer, noisy-input fidelity term, and mean-brightness constraint.

## Salt-and-pepper stress test

A paired test on 100 new reproducible 128x128 crops replaced 10% of whole RGB pixels:
5% with black and 5% with white. The realized corruption fraction was 9.997%. Both
learned-guidance methods retained their 150-step objectives and archive bands.

| Method | Mean PSNR (dB) | Mean SSIM |
|---|---:|---:|
| Noisy input | 15.1145 | 0.24952 |
| Median 3x3 | **29.3149** | **0.83146** |
| Median 5x5 | 27.0803 | 0.69990 |
| NIMA technical guidance | 20.5522 | 0.40601 |
| Archive discriminator guidance | 20.3323 | 0.39836 |
| Gaussian-SURE proxy (invalid noise model) | 23.0197 | 0.52602 |
| Clean-reference three-band oracle | 24.1414 | 0.59924 |

The 3x3 median filter won on 97 crops and the 5x5 median on three. Even the clean-target
oracle over the complementary three-band coefficient grid trailed the 3x3 median by
5.17 dB, showing that the three-band representation itself is poorly matched to impulse
noise. Gaussian SURE is not valid for salt-and-pepper noise; its row is only a deliberate
model-mismatch diagnostic using `sigma_proxy=sqrt(0.1)/2`. The original Self-FuseNet
paper is a super-resolution method and was not mislabeled as a salt-and-pepper denoiser.
Exact crops, all 800 saved PNGs, coefficients, and per-image metrics are under
`salt_pepper_100/`.

Both archived checkpoints were copied into the project and hash-verified:

- `checkpoints/netD_Clean_epoch_50.pth` — 22,264,366 bytes; SHA-256
  `30A40FC19BFC023E7D4F004FA567E81F41364FE5EB936F2ECE7E1AA708FEDF7F`
- `checkpoints/netD_Clean_epoch_100.pth` — 22,264,410 bytes; SHA-256
  `709F5C86DA2D7DB46C7FF8224D3A642884937CD8AE115DE1ECEA99186776FDB7`

## Limits and next validation

- All results are pilot subsets and random crops, with one noise realization per crop.
  Repeat several seeds and evaluate all 68 CBSD68 and 24 Kodak24 images before choosing
  a final algorithm.
- Only AWGN at one noise level was tested. Clipped, Poisson, impulse, and real sensor
  noise need separate protocols; standard SURE assumptions do not transfer automatically.
- Patchwise SURE uses independent 96x96 blocks without overlap-add, so seams may occur.
- BRISQUE/NIQE can be gamed without a risk constraint. Their constrained performance
  must be replicated on larger holdouts.
- The checkpoint pilot needs a longer, tuned optimization and comparisons under matched
  compute before it can be ranked against other coefficient-selection methods.
- SSIM/PSNR use clean targets only for reporting and labeled oracle diagnostics.

See `summary.csv`, `search_comparison.csv`, `per_image_results.csv`, `nr_summary.csv`,
and the method-specific output folders for raw results. Status and exact setup are in
`method_status.json`.
