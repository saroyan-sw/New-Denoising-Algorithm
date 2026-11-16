# KAN noise modeling with image quality: architecture review and preflight

Date: 2026-11-16. This is a new protocol; the completed scalar KAN and classical-denoising studies remain unchanged.

Subsequent result: the fixed natural-photo calibration passed the noise-ordering gate. See NATURAL_IMAGES_AND_REVISED_ARCHITECTURE.md. The original toy-domain result below remains valid; the metric is eligible only for a controlled auxiliary-loss test on the calibrated natural-photo domain.

## Confirmed setting

- Only noisy observations are available to training.
- One noisy observation per underlying scene.
- Independent zero-mean Gaussian pixels first.
- One shared unknown noise strength for the collection.
- Synthetic clean arrays may be used by the evaluation script, never as training targets.

## Two issues with the literal proposal

Let y=x+n and candidate denoised images be x_k=y-G(u_k). If the fidelity loss compares them with the noisy input, then mean_k ||y-x_k||² = mean_k ||G(u_k)||². The optimum is zero generated noise and unchanged output. This holds even when the generator is image-conditioned. A signed average error can instead cancel for arbitrary zero-mean candidates.

Even a perfect shared noise-law generator does not reveal the particular noise realization in y. For fresh independent noise eta with the correct zero mean and variance, E[(y-eta-x)²]=2 sigma². Averaging K such candidates gives sigma²(1+1/K), tending back to the original noisy image rather than a clean image.

These statements concern samples from the noise prior. Samples from a correctly inferred posterior conditioned on the observed pixel and image structure can behave differently.

## Actual mechanical and MUSIQ checks

The preflight uses 64×64 black, gray, step, and block images, 128 diagnostic repetitions, and shared sigma=0.1 for the mechanical control. The known sigma is only a control/evaluation setting; the prototype does not receive it.

On the step image, noisy MSE was 0.010018. Subtracting one independent oracle Gaussian sample increased it to 0.020036. Averaging 64 independent candidate subtractions gave 0.010173. The noisy-input fidelity loss always prefers a zero-amplitude generator.

The cached PyIQA 0.1.16 KonIQ MUSIQ model was frozen and its weight hash verified. Higher score is better. These are diagnostic scores, not optimization results:

| Pattern | Clean evaluation image | Gaussian sigma 0.05 | Gaussian sigma 0.1 |
| --- | ---: | ---: | ---: |
| Black | 19.084 | 20.592 | 22.002 |
| Gray | 17.654 | 33.616 | 33.616 |
| Step | 24.312 | 26.321 | 24.754 |
| Blocks | 37.766 | 42.597 | 42.966 |

MUSIQ failed the clean-versus-added-noise ordering on these toy scenes. Therefore the first training objective should have MUSIQ weight zero. This does not establish a general failure of MUSIQ on photographs: the toy scene content, resolution, grayscale replication, and preprocessing differ from its training distribution.

Raw data and MSE calculations are unclipped. Only the IQA input copy is clipped to [0,1] and replicated to RGB; the clipped fraction is recorded for every score. Black-plus-Gaussian has many negative values, so its IQA scores also include clipping effects. The clean gray versus noisy gray failure is present without that black-image issue.

## Recommended first architecture

1. Use a small masked CNN encoder and decoder to predict held-out pixels from surrounding spatial structure.
2. Start with one global spline KAN G_theta(u), shared by all images. Per-image conditioning is unnecessary when the underlying noise law is shared. The encoder context can later condition a KAN if the experiment changes to varying or signal-dependent noise.
3. Train the image branch with a held-out-pixel loss, using only noisy targets:
   L_J = mean over i in J of (D(y with J removed)_i - y_i)².
   All context/embedding paths must exclude those same held-out observations. A context code from the unmasked image would leak the target and invalidate the construction.
4. Treat the KAN as a noise prior, not an arbitrary independently sampled image residual to subtract uniformly.
5. Keep frozen MUSIQ as a reported diagnostic for these toy images. Introduce a weak quality regularizer only after its ordering/fidelity behavior has been validated in the intended domain.

For independent zero-mean noise and a center-excluding predictor, the expected held-out loss equals clean reconstruction MSE plus sigma². This is the Noise2Self rationale. The elementary neighbor predictor control measured that difference at approximately 0.01 for all four patterns.

### How generated samples could actually help reconstruction

To preserve the multiple-candidate idea, sample n_k from the learned shared prior, form x_k=y-n_k, and weight candidates by a learned spatial clean-image prior:

w_k proportional to p_X(y-n_k | surrounding context);
x_hat = sum_k w_k x_k / sum_k w_k.

This is posterior importance sampling when the proposal is the noise prior. The weights must depend on compatibility with image structure; uniform averaging does not perform the required inference. Apply it per pixel or small patch initially, since whole-image importance sampling can degenerate in high dimensions.

A practical later version predicts a clean-context mean and variance from held-out neighbors. With a fixed learned noise prior, a Monte Carlo convolution likelihood can train those parameters from noisy pixels. The Gaussian case has an analytic posterior baseline. Its noise variance must first be independently anchored from suitable noisy flat-region statistics; learning unconstrained clean and noise variances jointly only identifies their sum.

### Learning the shared noise prior remains an identification step

Residuals y-D(y) contain both noise and reconstruction error. Their distribution cannot simply be labeled the true noise distribution. For these simple scenes, begin with flat-region statistics estimated from noisy observations, compare against a Gaussian fit, and quantify selection/prediction bias using evaluation-only ground truth. A constant-signal collection is the easiest first identification case.

Do not give clean images, true residuals, or the true sigma to the trainable KAN. Do not make a physical noise-mechanism recovery claim from output histogram matching.

## Prototype that exists now

models.py implements a masked 2→16→16 CNN, four-dimensional pooled image context, a spatial decoder, and a genuine spline KAN. The shared prior is the default, using widths [1,8,8,1], grid 32, cubic splines. An explicit condition_noise=True option retains the proposed [5,8,8,1] conditional variant for later comparison.

The preflight verified forward propagation, finite gradients, one parameter update in each branch, exact invariance to changing every held-out target value, and checkpoint save/load. No clean reference was passed to either branch's wiring check. The initial conditional wiring result is also retained.

These are prototype checks, not a trained hybrid or a calibrated noise estimate. In particular, the prototype's residual-W2 step is only a differentiability test and does not establish that its residual distribution is noise.

## Staged experiment and gates

| Stage | Action | Acceptance / reason to stop |
| --- | --- | --- |
| 0, completed | Literal-objective controls, MUSIQ ordering, prototype wiring, checkpoint | Demonstrated loss collapse and metric mismatch; do not train the literal loss |
| 1, planned | Black/gray and simple patterns; one fixed observation per distinct scene; masked CNN baseline; center-excluding linear control | Held-out noisy validation improves; clean evaluation confirms denoising and edge preservation |
| 2, planned | Fit shared Gaussian and shared KAN noise priors from noisy flat-region evidence | Compare noise moments/W2, residual bias, autocorrelation, and held-out results; no true-noise training targets |
| 3, planned | Add posterior weighting/inference with the fixed noise prior | Beat uniform subtraction and match a correct Gaussian posterior control; report candidate effective sample size |
| 4, gated off here | MUSIQ regularizer ablation with frozen weights | Requires a suitable metric-validation domain; toy ranking failed |
| 5, later | Vary noise strength between images, then correlated noise | Only then test whether conditional embeddings add value |

Use independent scene identities and noise seeds for train, noisy validation, and clean evaluation. Reuse one fixed noisy observation per scene during training; do not silently substitute repeated independent noisy copies. Clean evaluation data must be inaccessible to the training loader.

## Sources

- [Noise2Self, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html): independence assumptions, held-out prediction, and noisy-only risk.
- [MUSIQ paper and official implementation](https://github.com/google-research/google-research/tree/master/musiq).
- [Google Research's description of MUSIQ](https://research.google/blog/musiq-assessing-image-aesthetic-and-technical-quality-with-multi-scale-transformers/): subjective aesthetic/technical quality assessment.
- Existing project MODEL_AND_METRIC_PROVENANCE.md records the exact PyIQA checkpoint and preprocessing.
