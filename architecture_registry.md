# Architecture registry

This file freezes the denoiser architectures considered for the bounded research
session. It separates the models that are scheduled now from higher-capacity models
that may be tested only if the stated development-only evidence appears. The final
holdout must never be used to choose among these architectures.

## Shared input, operator bank, and output rule

- Input: one RGB float32 noisy image `y` in `[approximately -0.4, 1.4]`. The input is
  not clipped before denoising. The known AWGN standard deviation is `25/255`.
- Initial compact bank: identity, bilateral, NLM with `h = 0.9 * sigma`, and color
  BM3D. The controlled A1r revision changes only NLM to the development-selected
  `h = 0.7 * sigma`; all later deployed candidates use that revised fixed order.
- Every fusion architecture combines fixed operator outputs `o_k(y)`. It does not
  update an operator or use a clean training target.
- The fused result is clipped to `[0,1]` only for metrics and saved images.
- Training objective: MC-SURE divided by `sigma^2`, minus `lambda` times standardized
  frozen MUSIQ (KonIQ checkpoint) score. MUSIQ parameters remain frozen; gradients
  pass only from its input to fusion parameters. PSNR selects among trained candidates;
  DISTS is evaluation-only.
- Development data: 16 source-disjoint training crops and the immutable 20-image
  benchmark in `benchmark20.json`. Three fixed benchmark noise draws are used.

## A0 — individual operators (scheduled baselines)

No learned parameters. Test identity, Gaussian sigma 0.7 and 1.1, bilateral,
median 3x3, NLM with `h/sigma` in `{0.7, 0.9, 1.1}`, and color BM3D. These establish
the strongest classical comparator and operator error diversity. Official pretrained
color DnCNN-blind and DRUNet-known-sigma are compatible neural comparators, not fusion
members.

Commands:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py classical --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py neural --split benchmark
```

## A1 — compact global parallel convex fusion (scheduled; primary model)

One trainable logit per compact-bank operator. For every pixel and channel:

```text
w = softmax(a)                         # four nonnegative weights, sum exactly one
x_hat = sum_k w[k] * o_k(y)            # same weights for every pixel and image
```

Parameter count: 4 logits, with only 3 identifiable degrees of freedom. Initial
weights are the exact constrained MC-SURE solution on the 16 training crops.

Scheduled variants:

| ID | lambda | purpose |
|---|---:|---|
| `v3_lambda_0p0` | 0.00 | fidelity-only control; exact convex SURE weights |
| `v3_lambda_0p01` | 0.01 | weak frozen-MUSIQ regularization |
| `v3_lambda_0p03` | 0.03 | stronger frozen-MUSIQ regularization |

For positive lambda: Adam, learning rate 0.08, 40 updates, deterministic image
cycling, actual optimizer/RNG checkpoint recovery, no scheduler, no mixed precision.
An equal-weight A1 model is also evaluated as a non-trained control.

Commands:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py sure
.venv\Scripts\python.exe scripts\research_session_v3.py oracle
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.01
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.03
```

The reference-assisted convex oracle is a diagnostic ceiling only. It is fitted and
reported on the development benchmark and can never be deployed or evaluated on the
final holdout.

## A1r — NLM-selected compact fusion (scheduled evidence-based revision)

The predeclared NLM sweep found `h=0.7*sigma` at 29.0572 dB versus 28.3590 dB for
the original `h=0.9*sigma`. A validation-only oracle with the one-operator substitution
increased global headroom over CBM3D from 0.0068 to 0.0140 dB while per-image-over-
global headroom stayed only 0.0289 dB (one of 20 images at least 0.1 dB). Therefore,
repeat the same three A1 lambdas as `v4_lambda_0p0`, `v4_lambda_0p01`, and
`v4_lambda_0p03`, changing no other model or optimizer setting. These bring the total
run count to 8/10 including the two ineligible pilots. A2-A4 remain closed.

## A2 — feature-conditioned spatial convex fusion (gated, not scheduled yet)

Test only if A1's development oracle has useful per-image/pixel headroom and visual
inspection shows different operators winning in different spatial regions. Fixed
features are computed from the noisy image: RGB (3 channels), luminance, luminance
local 3x3 mean, luminance local 3x3 standard deviation, and luminance Sobel gradient
magnitude (7 channels total). A small
gating network predicts four per-pixel logits:

```text
Conv(7, 16, 3, padding=1) -> SiLU
Conv(16, 16, 3, padding=1) -> SiLU
Conv(16, 4, 1) -> softmax over operators
x_hat[p] = sum_k w[p,k] * o_k(y)[p]
```

Parameter count: 3,412 including biases. Add total variation on the weight maps with
coefficient `1e-3`; otherwise use the same MC-SURE/MUSIQ objective and data. Initialize
the last-layer biases from the selected A1 logits and all other weights with Kaiming
uniform initialization. Maximum one fidelity-only control and one positive-lambda run.

Gate to open A2 (all must hold on development data):

1. Per-image reference oracle exceeds the global oracle by at least 0.10 dB mean PSNR.
2. At least 5 of 20 inspected images show spatially distinct operator strengths.
3. A1 does not already establish its improvement claim.

## A3 — two-scale spatial convex fusion (gated after A2)

Test only if A2 visibly leaves scale-dependent residual noise or texture loss. Use
the same A2 gating network independently at native scale and at 1/2 resolution.
The half-resolution branch receives area-downsampled noisy/operator images and is
bilinearly upsampled with `align_corners=False`. A learned global two-logit softmax
convexly combines the native and upsampled estimates.

```text
x_native = A2(y, operators(y))
x_half   = upsample(A2(down2(y), down2(operators(y))))
q = softmax(b)                           # two global scale weights
x_hat = q[0] * x_native + q[1] * x_half
```

The two A2 branches share parameters. Added parameter count: 2 logits. Initialize
from A2 with scale weights `(0.8, 0.2)`. Maximum one run, using the best development
lambda already selected for A2; do not retune lambda.

Gate to open A3 (all must hold):

1. A2 improves A1 primary PSNR but has visible coarse residual noise on at least 4
   benchmark images, or the multiscale reference oracle adds at least 0.10 dB.
2. Remaining run and time budget permit a complete run plus all-20 visual review.

## A4 — depth-two sequential residual refinement (last gated option)

Test only if parallel convex fusion consistently leaves structured residuals that a
second operator pass removes in a development-only diagnostic. Stage 1 is the selected
A1 or A2 estimate. Stage 2 applies only identity, NLM-0.9, and color BM3D to the stage-1
estimate at the same known sigma and forms another convex blend. A single learned gate
`rho = sigmoid(r)` limits the correction:

```text
z1 = selected parallel fusion(y)
z2 = sum_j softmax(c)[j] * operator_j(z1)
x_hat = (1-rho) * z1 + rho * z2
```

Parameter count over the selected stage-1 model: 4 logits (3 stage-2 logits plus
`rho`). Initialize stage 2 to identity and `rho=0.1`. Freeze stage 1 for the first
10 updates, then jointly optimize for at most 30 more updates. Maximum depth is two;
no recursive or unrestricted sequence is permitted. Maximum one run using the already
selected lambda.

Gate to open A4 (all must hold):

1. A diagnostic second pass improves the development reference oracle by at least
   0.10 dB over the selected parallel architecture.
2. Inspection identifies structured residual noise on at least 4 images without a
   consistent texture-erasure failure.
3. Remaining run/time budget permits full evaluation and visual review.

## Stop and selection rules

- A2-A4 are pre-registered possibilities, not permission to run them automatically.
  If their gates are not met, record that and stop at A1.
- Count every started substantial training run, including failed reruns. The overall
  cap remains 10; two preliminary ineligible runs already count.
- At battery `<=15%`, save an atomic checkpoint and stop. Respect the cumulative
  four-hour active-work budget across resumes.
- Inspect four five-image comparison sheets plus the worst-delta panel after every
  meaningful candidate. Document blur, residual noise, ringing, color shifts, and
  structure loss before choosing the next architecture.
- Lock the selected model/configuration before acquiring Urban100 081-100. The final
  holdout is evaluated exactly once and cannot reopen architecture selection.

Exact current resume order and cache-aware commands are maintained in
`next_steps.md`; the executable A0/A1 implementation is `scripts/research_session_v3.py`.
