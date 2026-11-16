# Natural-image check and revised architecture — 2026-11-16

## What the earlier experiment actually did

The earlier stage was preflight, not trained denoising:

1. Used 64×64 black, gray, step, and block images to test independent oracle Gaussian subtraction and candidate averaging.
2. Evaluated 16 frozen MUSIQ scores on those toy images and their noisy/flat variants.
3. Implemented a masked CNN plus a shared spline KAN and checked forward propagation, gradients, one update per branch, exclusion of hidden pixels, and checkpoint recovery.

The subtraction control used sigma=0.1; the illustrative MUSIQ table uses sigma=0.05. Clean synthetic arrays were used only for evaluation. The illustrated thumbnails are conceptual, not saved model reconstructions.

The measured step-image MSE was 0.010018 before subtraction, 0.020036 after subtracting one independent oracle Gaussian sample, and 0.010173 after averaging 64 candidates. These numbers did not come from a trained KAN or denoiser.

The generated explanation is results/preflight_explainer.png. Its generation/edit prompt and provenance are in results/preflight_explainer_prompt.json.

## Fixed natural-image protocol

Before scoring, fixed four built-in scikit-image 0.26.0 photographs: astronaut, coffee, chelsea (cat), and camera. Their original sizes were retained: 512×512, 400×600, 300×451, and 512×512; the camera image was replicated into RGB. No image was selected based on its MUSIQ result.

Evaluated added iid Gaussian noise at standard deviations 0.025, 0.05, and 0.1, with three fixed noise seeds per image. Also evaluated a Gaussian-smoothed noisy image, independent-noise subtraction, a blurred reference, and flat mean-color output. Total: 72 saved scores. This is metric calibration, not a fresh held-out evaluation of a trained denoiser.

The same frozen PyIQA 0.1.16 KonIQ MUSIQ checkpoint was reused, with SHA-256 e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa. Raw arrays and MSE calculations were unclipped; only copies sent to MUSIQ/display were clipped. The clipped fraction is recorded for every score.

## Measured results

MUSIQ is higher-is-better. Noise columns are means over three seeds:

| Photograph | Unmodified reference | Noise 0.025 | Noise 0.05 | Noise 0.1 |
| --- | ---: | ---: | ---: | ---: |
| Astronaut | 78.46 | 76.99 | 70.47 | 54.17 |
| Coffee | 75.09 | 70.15 | 60.92 | 52.78 |
| Cat | 72.50 | 70.66 | 66.85 | 59.38 |
| Camera | 67.07 | 64.42 | 55.55 | 45.50 |

All four mean curves were strictly ordered by added noise. All mean-color flat outputs also scored lower than their references. The predeclared gate for a small auxiliary-quality-loss ablation passed.

Independent-noise subtraction at sigma=0.05 approximately doubled raw distortion to the reference and reduced mean MUSIQ relative to the noisy input on every photo. Thus the random-subtraction issue persists even when the metric behaves sensibly.

Gaussian smoothing reduced reference MSE but also reduced MUSIQ relative to the noisy input for all four images. This shows a noise/detail trade-off: sensible ordering by noise level does not make the quality score identical to pixel fidelity.

See results/natural_image_comparison.png for real input/output examples, results/natural_musiq_curve.png for the score curves, and results/natural_scores.json for all rows.

## Revised architecture

The image branch models scene structure; the shared KAN models the noise prior:

~~~mermaid
flowchart LR
    Y[Noisy RGB image] --> M[Hide target pixels]
    M --> E[CNN encoder and spatial decoder]
    E --> P[Prediction / clean-context prior]
    U[Uniform samples] --> G[Shared spline KAN]
    G --> N[Noise-prior samples]
    Y --> C[Candidate images: y minus n]
    N --> C
    C --> I[Structure-conditioned posterior inference]
    P --> I
    I --> X[Denoised estimate]
    X --> Q[Frozen MUSIQ: auxiliary score]
~~~

For the confirmed shared unknown Gaussian level, begin with one global G(u). An image-specific KAN code is a later ablation for varying or signal-dependent noise. The CNN can use a four-dimensional pooled context plus spatial features for reconstruction; every image-dependent path in the blind prediction must exclude the held-out targets.

MUSIQ supplies an optional additional training signal on the restored image. It does not replace the observation model, estimate the realized noise, or make noisy-input reconstruction MSE valid. Comparing y against y-G(u) still reduces exactly to penalizing G(u)² and encourages zero subtraction.

Train in stages:

1. Masked RGB denoiser with noisy held-out targets only; establish a lambda=0 quality-loss baseline.
2. Learn/estimate the shared noise prior from independent noisy calibration images or suitable flat-region evidence. Compare the KAN with a Gaussian fit using the same evidence. Reconstruction residuals require bias/selection diagnostics and cannot be assumed to equal true noise.
3. Freeze/anchor that noise prior and add structure-conditioned posterior inference. The Gaussian case should have an analytic posterior control. Uniform averaging of prior samples is the negative control.
4. Compare lambda=0 against a small standardized frozen-MUSIQ term. Choose settings using noisy validation criteria; keep clean reference arrays exclusively in final evaluation. Check edge/detail preservation, MSE, noise moments, tails, and whether score improvement corresponds to actual recovery.

The held-out noisy-pixel MSE is valid for the blind predictor under the independence assumptions. Once a posterior refinement reuses the observed center pixel, that final output is no longer blind; its fidelity needs an appropriate likelihood or a correctly implemented SURE-style criterion using a separately estimated noise level. Do not claim the blind-pixel risk identity for a center-pixel bypass.

Sample-based posterior weighting must use an explicit spatial clean-image prior (and be diagnosed for effective sample size), not merely the assumption that independent noise samples cancel the observed noise. Whole-image importance sampling is likely inefficient; start per pixel or small patch.

## Implementation status

models.py now supports RGB as well as grayscale. The RGB prototype has 9,271 parameters, including the shared 2,960-parameter KAN. Forward shape, held-out target invariance, finite gradients, one parameter update per branch, and checkpoint save/load passed. A separate MUSIQ test verified a finite nonzero gradient with respect to a natural-image input while all metric weights stayed frozen.

These remain wiring/metric checks. No full hybrid has been trained, no posterior coupling has been implemented yet, and no denoising improvement of the proposed learned architecture is claimed.

## Image credits

Samples come from the installed scikit-image data collection. Its local documentation identifies the astronaut photograph as NASA/public domain; coffee as CC0 by Rachel Michetti, courtesy of Pikolo Espresso Bar; chelsea as CC0 by Stefan van der Walt; and camera as CC0 by Lav Varshney.
