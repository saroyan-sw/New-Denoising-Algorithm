# Benchmark and Naturalness-Prior Strategy

## Decision summary

The first benchmark should be **color additive white Gaussian noise (AWGN) at a known, fixed level**, beginning with \(\sigma=25/255\), on **CBSD68 and Kodak24**. Development and any clean-prior fitting should use disjoint clean images, initially DIV2K train/validation. Clean references are used only for evaluation and oracle diagnostics, never by the proposed restoration optimizer.

This is deliberately not the most realistic task. It is the cleanest experiment for determining whether a deterministic or semi-deterministic mechanism, guided by a differentiable naturalness signal, has useful restoration capacity. AWGN remains a normal research benchmark: Restormer reports the standard \(\sigma=15,25,50\) protocol, CODE uses the same levels and CBSD68/Kodak24/McMaster, and the NTIRE 2025 denoising challenge used fixed \(\sigma=50\) AWGN. Sources: [Restormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf), [CODE, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Comprehensive_and_Delicate_An_Efficient_Transformer_for_Image_Restoration_CVPR_2023_paper.pdf), and [NTIRE 2025 denoising challenge](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/html/Sun_The_Tenth_NTIRE_2025_Image_Denoising_Challenge_Report_CVPRW_2025_paper.html).

The recommended progression is:

1. fixed known AWGN at \(\sigma=25\);
2. the full \(\sigma\in\{15,25,50\}\) non-blind protocol;
3. blind and spatially varying Gaussian noise;
4. SIDD/DND real sRGB noise.

Advancing should depend on evidence. If the method cannot improve consistently over bilateral/nonlocal-means filtering or approach CBM3D on fixed AWGN, a harder noise model will obscure rather than solve the problem.

## 1. Defining the denoising task

Let the clean RGB image be \(x\in[0,1]^{H\times W\times3}\) and the observation be

\[
y=x+n,\qquad n\sim\mathcal N(0,\sigma^2I).
\]

The proposed method receives \(y\), the restoration mechanism, and its frozen quality/prior models. It does not receive \(x\). In the non-blind setting it may receive \(\sigma\); in the blind setting it may not. The clean image is retained by the evaluation harness only.

### Comparison of candidate task definitions

| Setting | Scientific value | Advantages for this project | Main disadvantages | Recommended role |
|---|---|---|---|---|
| Fixed, known AWGN | Isolates restoration under an exact likelihood | Fast; deterministic; standard baselines; unlimited paired evaluation from clean images; known residual energy; easiest representational-ceiling test | Unrealistic after camera ISP; favors Gaussian-specific algorithms | **Primary first benchmark** |
| Multiple known AWGN levels | Tests robustness across weak/strong corruption | Standard \(15,25,50\) reporting; still simple; shows whether conclusions survive noise strength | Either separate tuning or a noise-level input is needed | Immediate extension after \(\sigma=25\) |
| Blind AWGN, one model/optimizer for \(\sigma\in[0,50]\) | Tests implicit noise assessment and adaptation | Closer to deployment; still permits exact clean evaluation; blind Restormer/SCUNet-style baselines exist | Confounds restoration capacity with noise estimation; fairness requires declaring which baselines receive \(\sigma\) | Second-stage benchmark |
| Spatially varying Gaussian noise | Tests whether local/patch parameters add value | Directly relevant to the project's local weighting; exact synthetic ground truth and noise map are possible | No single universally dominant protocol; a custom map reduces paper-to-paper comparability | Controlled stress test, not headline result |
| Poisson or Poisson–Gaussian noise | Better model of photon-limited raw sensing | Known likelihood; relevant to cameras/scientific imaging; self-supervised theory can apply | Parameter conventions vary; sRGB images no longer obey the physical model | Later likelihood-robustness test |
| Real sRGB camera noise | Tests practical output after an ISP | SIDD/DND are recognized; captures signal-dependent and spatially correlated artifacts | Noise, demosaicing, sharpening, compression, alignment, and camera domain are entangled; residual likelihood is unknown | Later external-validity benchmark |
| Unknown real noise in arbitrary Internet images | Broad practical ambition | No paired data needed at use time | Almost impossible to rank faithfully; no unambiguous ground truth; domain and corruption are uncontrolled | Qualitative demonstration only until core method works |

### Why fixed AWGN is the right first decision

The task gives us three things that the satellite setting did not:

- an exact observation model;
- clean references for evaluation without giving them to the optimizer;
- widely published reference points for classical, supervised, and self-supervised methods.

It also permits a decisive **oracle ceiling experiment**: optimize the same restoration parameters against the hidden clean image. If even oracle optimization performs poorly, the restoration representation—not the naturalness objective—is the bottleneck.

## 2. Dataset selection

### Recommended small dataset set

| Dataset | Contents and size | Resolution | Clean/noisy and pairing | Conventional use | Access and limitations | Decision |
|---|---|---|---|---|---|---|
| DIV2K | 1,000 diverse high-quality RGB images: 800 train, 100 validation, 100 test | Approximately 2K | Clean images; synthetic noise is generated on demand, so pairs are exact | Clean training corpus for modern restoration; part of the common DFWB Gaussian-denoising recipe | Academic-research license; source-image copyrights remain with owners. [Official page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) | Use 800 train for fitting a prior, 100 validation for all decisions |
| CBSD68 | 68 color natural photographs drawn from the Berkeley set | 481×321 or 321×481 | Clean originals; AWGN synthesized exactly | Standard color Gaussian-denoising test set at \(\sigma=15,25,50\) | Small and relatively low-resolution; research redistribution is common but should be attributed to BSDS. [Berkeley source](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/index2) | Primary test set |
| Kodak24 | 24 lossless 24-bit color photographs | 768×512 or 512×768 | Clean originals; AWGN synthesized exactly | Standard color restoration/denoising test set | Very small; the hosting page describes unrestricted usage but is not a modern formal dataset license. [Image suite](https://r0k.us/graphics/kodak/index.html) | Independent secondary test set |
| BSD68 | Grayscale form of 68 Berkeley images | 481×321 or 321×481 | Clean originals with synthetic noise | Long-standing grayscale Gaussian benchmark | Same small/low-resolution limitation; color is more relevant to this project | Optional compatibility result only |
| Urban100 | 100 high-resolution urban/architectural images | Variable, often around megapixel scale | Clean originals with synthetic noise | Common structured-detail stress test | Narrow content distribution; more common in super-resolution, though also used for Gaussian denoising | Optional edge/texture stress set |

Modern high-capacity denoisers commonly train on a larger combination: 800 DIV2K + 2,650 Flickr2K + 400 BSD images + 4,744 Waterloo Exploration Database images. Restormer's supplement documents this 8,594-image recipe, while SwinIR publishes the same train/test organization. [Restormer supplement](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Zamir_Restormer_Efficient_Transformer_CVPR_2022_supplemental.pdf), [SwinIR repository](https://github.com/JingyunLiang/SwinIR).

We should **not** begin by recreating that full training corpus. Our restoration mechanism is trained per image or has few parameters. DIV2K alone is large enough to fit/calibrate a clean prior and small enough to audit. Published pretrained supervised baselines may retain their original training data, but this must be reported so that the comparison is interpreted as a performance reference rather than a matched-data contest.

### Real-noise datasets reserved for later

| Dataset | Contents and scale | Pairing and noise | Normal use | Access limitation | Decision |
|---|---|---|---|---|---|
| SIDD | About 30,000 captures from 10 scenes, five smartphones, multiple lighting/exposure settings; SIDD Medium has 320 noisy/GT pairs | Real rawRGB and rendered sRGB with estimated ground truth; SIDD benchmark has 40 held-out images evaluated on 32 256×256 blocks each | Training (Medium/Full), validation, and benchmark testing; dominant smartphone denoising dataset | MIT license; official evaluation currently points to Kaggle because the original server is down. [Official SIDD page](https://abdokamel.github.io/sidd/) | First real-noise dataset after Gaussian success |
| DND | 50 high-resolution real scenes from four consumer cameras; 20 crops per image, 1,000 evaluated crops | High-ISO noisy observations with carefully registered low-ISO references; benchmark references remain server-side; raw and sRGB tracks | Test-only cross-camera benchmark | Non-commercial license and online submission; too small for ordinary training. [Official DND description](https://noise.visinf.tu-darmstadt.de/), [benchmark protocol](https://noise.visinf.tu-darmstadt.de/benchmark/) | Cross-dataset test after SIDD; not a development set |
| MIDD | 400,000 noisy photos and 20,000 corresponding ground-truth images across mobile sensors and varied scenes/lighting | Real paired mobile data | Newer large-scale real-noise training/evaluation resource | Much newer and less standardized than SIDD/DND | Track as a future generalization set, not the initial benchmark |

SIDD and DND remain active in supervised and self-supervised literature: Restormer trains on SIDD and tests on both; AP-BSN, spatially adaptive self-supervision, and ICCV 2023 unpaired work report them as principal real-noise benchmarks. [Restormer](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf), [AP-BSN](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.pdf), [spatially adaptive self-supervision](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Spatially_Adaptive_Self-Supervised_Learning_for_Real-World_Image_Denoising_CVPR_2023_paper.pdf).

### Datasets for a quality model, not the denoiser

If a quality head is trained or calibrated, its data must be kept separate from restoration test images.

- **KADID-10k:** 81 pristine images, 25 synthetic distortion types, five levels, 10,125 distorted images with crowdsourced ratings. It is useful for controlled distortion sensitivity but has limited source content. [Official database](https://database.mmsp-kn.de/kadid-10k-database.html)
- **PIPAL:** 250 references and roughly 29,000 outputs, including traditional, deep, and GAN restoration distortions with more than one million pairwise judgments. It is unusually relevant to restoration artifacts. [PIPAL, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1421_ECCV_2020_paper.php)
- **KonIQ-10k:** 10,073 authentically distorted Internet photographs with crowdsourced MOS. It offers content diversity but no clean reference. [KonIQ-10k paper](https://arxiv.org/abs/1910.06180)
- **SPAQ:** 11,125 photographs from 66 smartphones with overall quality plus noisiness, sharpness, color, brightness, and contrast opinions. It is valuable if we need a noise-specific quality head. [SPAQ, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html)

These databases teach human-perceived quality, not source-faithful denoising. They should not overlap with the denoising evaluation set and cannot substitute for PSNR/SSIM against known clean images.

## 3. Baseline evaluation protocol

### Canonical first protocol

1. Decode RGB images without resizing or JPEG recompression.
2. Convert to float in \([0,1]\).
3. Generate \(y=x+n\), \(n\sim\mathcal N(0,(25/255)^2I)\), using a versioned image-specific seed.
4. Store the noise arrays or their hashes. Explicitly declare whether the noisy observation is clipped; the recommended paper-comparable convention is no clipping before denoising and clipping the final estimate to \([0,1]\) for evaluation.
5. Tune every proposed hyperparameter on DIV2K validation only.
6. Lock the configuration, then evaluate CBSD68 and Kodak24 once.
7. Report per-image values, mean, median, standard deviation, and a paired bootstrap 95% confidence interval relative to CBM3D and the strongest simple filter.
8. Repeat the final comparison across five noise seeds as a robustness appendix. Keep one canonical realization for direct reproducibility.

When \(\sigma=25\) is stable, rerun unchanged logic at \(15\) and \(50\). Any per-level retuning must be declared.

### Baseline ladder

| Tier | Methods | Question answered |
|---|---|---|
| Sanity | Noisy input/identity; Gaussian blur; median; bilateral; non-local means | Does the proposed optimization beat trivial smoothing and avoid harming easy images? |
| Strong classical | CBM3D/BM3D; optionally WNNM if a maintained implementation is verified | Does it approach a respected non-neural denoiser under its correct AWGN assumptions? |
| Established supervised | DnCNN; FFDNet; DRUNet | What is the gap to classic learned AWGN denoisers? DRUNet/FFDNet receive the true noise map only in the explicitly non-blind track. |
| Modern supervised reference | Restormer or SwinIR pretrained non-blind and blind models | What is the gap to a high-capacity modern denoiser? This is a reference ceiling, not a matched compute/data comparison. |
| No-clean-target learning | Noise2Self/Noise2Void; Neighbor2Neighbor; one single-image method such as Self2Self or a verified zero-shot blind-spot implementation | How far are we from recognized self-supervised and per-image optimization approaches? |
| Project methods | Fixed deterministic pipeline; corrected differentiable fusion; proposed quality objective; clean-target oracle using the identical parameterization | Is improvement due to optimization, and is the remaining gap caused by the objective or the representation? |

Noise2Self formalizes denoising through self-supervision, Neighbor2Neighbor trains from noisy images through neighbor subsampling, and recent zero-shot blind-spot work remains active at NeurIPS 2025. [Noise2Self, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html), [Neighbor2Neighbor, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html), [zero-shot blind-spot denoising, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html).

### Metrics and their roles

| Measure | Use | Weakness |
|---|---|---|
| PSNR/MSE | Primary fidelity measure for synthetic pairs; exact and comparable | Favors conditional averages and can prefer oversmoothing |
| SSIM | Primary complementary structural measure | Local statistic, parameter-sensitive, still imperfect for texture/perception |
| MAE | Robust, interpretable pixel error | Not perceptually calibrated |
| LPIPS or DISTS | Secondary full-reference perceptual distance | Learned-domain bias; differentiable metrics are attackable; not available at real test time without a reference |
| NR-IQA ensemble | Diagnostic perceptual/naturalness reporting only | Can prefer metric-specific artifacts and should never be the sole evidence for a method that optimizes it |
| Residual diagnostics | Residual mean/variance, autocorrelation, spectral flatness, edge/texture-stratified error | A white residual can still contain removed signal; use only with fidelity metrics |
| Runtime/resources | CPU/GPU time, optimization steps, memory, learned/prior parameters | Hardware-dependent; report environment |

The perception–distortion trade-off means that a method can improve perceptual realism while worsening source fidelity; neither PSNR nor a naturalness score alone defines success. [Blau and Michaeli, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html).

Qualitative figures should use predeclared images and crops, always showing clean, noisy, proposed, CBM3D, one self-supervised method, and one modern supervised reference at identical zoom and intensity scaling.

## 4. Candidate differentiable naturalness and quality signals

The crucial distinction is:

- **fidelity:** whether \(\hat x\) matches the unknown source \(x\);
- **naturalness:** whether \(\hat x\) lies near a clean-image distribution;
- **perceptual quality:** whether humans prefer its appearance;
- **observation consistency:** whether \(\hat x\) plausibly generated \(y\) under the noise model.

No no-reference model supplies all four.

### Direction comparison

| Direction | Signal and differentiability | Training/data | What it measures | Exploitation/failure mode | Cost and suitability |
|---|---|---|---|---|---|
| Clean-vs-degraded discriminator | Scalar/logit or patch map; fully differentiable | Requires clean images and representative degradations; no paired clean/noisy instances are necessary | Separability of the two training distributions | Learns degradation shortcuts; rewards contrast/sharpening; direct ascent finds adversarial textures; output may be natural but unrelated to input | Cheap–moderate; suitable as one experimental prior, not the default truth |
| Pretrained NR-IQA model (MUSIQ, LIQE, Re-IQA, TOPIQ-like) | MOS/quality prediction; normally differentiable through the image after verifying preprocessing | No new training with released weights; optional domain calibration using MOS datasets | Human-perceived quality/naturalness on its training distribution | Patch/crop and content bias; can reward smoothing or aesthetics; white-box optimization can attack it | Moderate; strongest low-effort first candidate, preferably an ensemble |
| CLIP/DINO-style encoder plus quality head | Frozen image representation and learned scalar/ranking head; differentiable | Small head needs PIPAL/KADID/KonIQ/SPAQ labels or synthetic ranking pairs | Whatever the head is trained to predict; generic encoders mainly encode content/semantics | Invariance to low-level noise can make gradients uninformative; a head can exploit content; direct optimization can target features | Moderate; useful as a controlled learned representation, not zero-shot naturalness by itself |
| Full-reference perceptual metric (LPIPS, DISTS) | Differentiable distance between two images | Pretrained; no new training | Perceptual similarity to a supplied reference | The only available reference at inference is noisy; minimizing distance to it preserves noise. Learned distances also admit adversarial perturbations | Moderate; excellent evaluation/oracle diagnostic, poor standalone NR objective |
| Clean feature-distribution statistics | Mahalanobis, Gaussian mixture, MMD, or patch-density score in a frozen encoder | Requires only a corpus of clean images; no quality labels or degraded pairs | Proximity to clean feature statistics | A one-image feature match can collapse to common textures/means; ignores source identity; covariance/domain mismatch | Low–moderate; highly aligned with the no-paired-target philosophy and worth testing |
| Learned denoiser prior / RED / plug-and-play | Vector field or proximal denoising step rather than a scalar score; gradients may be obtained by backpropagation | Requires a pretrained denoiser, commonly trained on synthetic pairs or through self-supervision | Implicit clean-image/noise prior | The prior may perform the actual denoising, making the deterministic mechanism scientifically redundant; fixed-point artifacts and domain bias | Moderate; valuable comparator, less clean as a test of our central idea |
| Diffusion or score-model prior | Approximation to \(\nabla_x\log p_t(x)\) across noise scales; differentiable or directly supplies a score | Large pretrained clean-image generative model; no task-paired data required | Natural-image probability/score, not source fidelity | Semantic hallucination, posterior-mode attraction, prior/domain dominance, unstable score-distillation gradients | Very high; scientifically interesting but inappropriate for the first fast benchmark |
| Classical NSS (NIQE/BRISQUE) | Hand-designed or learned score; some implementations are only partly differentiable | Pristine-feature model or labeled SVR | Deviation from natural scene statistics / predicted MOS | Strong domain bias; rewards oversmoothing/contrast changes; low-dimensional statistics are easy to satisfy | Low; retain as reporting/legacy ablation, not primary optimizer |

Relevant modern IQA models include [MUSIQ, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html), [LIQE, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.html), and [Re-IQA, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Saha_Re-IQA_Unsupervised_Learning_for_Image_Quality_Assessment_in_the_Wild_CVPR_2023_paper.html). LPIPS and DISTS provide differentiable full-reference perceptual comparisons. [LPIPS, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.pdf), [DISTS, TPAMI](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Ding20).

Diffusion priors are not merely speculative: DDRM uses a pretrained diffusion model for unsupervised posterior sampling in linear inverse problems, and later work continues to combine pretrained diffusion denoisers with observation guidance. Their computational and identifiability costs make them a later benchmark rather than the first objective. [DDRM, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html), [diffusion restoration with preconditioned guidance, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Garber_Image_Restoration_by_Denoising_Diffusion_Models_with_Iteratively_Preconditioned_Guidance_CVPR_2024_paper.html).

### Metric exploitation is an expected failure mode

Direct optimization turns an IQA model into a white-box adversarial target. Recent evaluations find that leading neural NR-IQA metrics can be manipulated rapidly, including at high resolution, and that high benchmark correlation does not imply adversarial robustness. [Shumitskaya et al., CVIU 2024](https://doi.org/10.1016/j.cviu.2023.103913), [Guardians of Image Quality, ICML 2025](https://openreview.net/forum?id=Y3EQLjoYdQ&noteId=cyoCfzcqph).

Every quality objective must therefore pass the following audit before it guides a restoration claim:

1. **Monotonicity:** rank clean, weak noise, strong noise, reasonable denoising, oversmoothing, sharpening, contrast shift, and synthetic texture in the expected order.
2. **Attack test:** optimize pixels directly for the score under an \(L_\infty\) or \(L_2\) budget. Reject or constrain models that generate invisible/visible score attacks easily.
3. **Transfer test:** optimize model A; measure PSNR, SSIM, DISTS/LPIPS, and held-out quality models B and C. A gain only on A is metric exploitation.
4. **Restoration constraint:** combine naturalness with observation consistency, not merely input similarity.
5. **Validation-only stopping:** select iteration count and objective weights on DIV2K validation; never select the best CBSD68/Kodak result.
6. **Restricted degrees of freedom:** initially optimize only the restoration mechanism's interpretable parameters, not free pixels.
7. **Randomized views:** average the objective over fixed multiscale crops/flips or mild differentiable resampling so a single patch/scale cannot dominate.
8. **Trajectory reporting:** save all iterations and show whether quality score continues rising after PSNR/visual fidelity begins falling.

For known AWGN, a safer composite objective is conceptually

\[
\mathcal L(\hat x;y)=
\lambda_Q\,Q(\hat x)
+\lambda_E\left(\frac{\|y-\hat x\|_2^2}{N}-\sigma^2\right)^2
+\lambda_W R_{\mathrm{white}}(y-\hat x)
+\lambda_C C(\hat x,y),
\]

where \(Q\) is naturalness, the second term is the AWGN discrepancy principle, \(R_{\mathrm{white}}\) penalizes structured residual correlation, and \(C\) is a weak content-preservation constraint. None uses the clean target. The exact terms and weights must be ablated; residual energy alone can remove signal, while a strong distance to \(y\) preserves noise.

## 5. Four concrete candidate setups

### Setup A — Fixed-AWGN mechanism test

- **Task:** RGB non-blind AWGN, initially \(\sigma=25\), then \(15,50\).
- **Prior data:** DIV2K train only if the naturalness model needs clean fitting; no denoiser training required.
- **Validation/test:** DIV2K validation / CBSD68 + Kodak24.
- **Baselines:** identity, Gaussian, median, bilateral, nonlocal means, CBM3D, DnCNN, FFDNet or DRUNet, Restormer/SwinIR, Noise2Self, Neighbor2Neighbor.
- **Quality model:** frozen ensemble of two architecturally different NR-IQA models, e.g. MUSIQ plus LIQE or Re-IQA; legacy discriminator included only as an ablation.
- **Objective:** naturalness + known-noise residual-energy/whiteness + weak content preservation.
- **Evaluation:** PSNR, SSIM, MAE, DISTS or LPIPS, residual statistics, runtime, paired confidence intervals.
- **Scientific question:** can the existing restoration parameterization and differentiable guidance beat simple filters and approach CBM3D when the inverse problem is completely controlled?

This is the recommended first experiment.

### Setup B — Naturalness-model shootout under an identical mechanism

- **Task/data/noise:** identical to Setup A, \(\sigma=25\) only.
- **Quality variants:** no learned quality term; archived discriminator; MUSIQ; LIQE/Re-IQA; their ensemble; frozen DINOv2 feature-statistics prior fitted on DIV2K; clean-target oracle.
- **Baselines:** the fixed mechanism without optimization and CBM3D are sufficient in addition to Setup A's stored outputs.
- **Objective:** identical consistency terms and optimization budget; only \(Q\) changes.
- **Evaluation:** ground-truth restoration metrics, cross-metric transfer matrix, gradient norms/frequency spectra, direct pixel attack test, and score-versus-PSNR trajectories.
- **Scientific question:** does a better naturalness model produce genuine restoration gains, or merely a more effective metric attack?

This setup isolates the central hypothesis more cleanly than changing the restoration architecture and objective simultaneously.

### Setup C — Blind and spatially varying Gaussian adaptation

- **Task:** RGB AWGN with imagewise unknown \(\sigma\sim U[5,50]\), followed by a fixed spatial map protocol.
- **Prior data:** DIV2K train if needed; noisy samples generated online without exposing \(\sigma\) to the proposed method.
- **Validation/test:** DIV2K validation / CBSD68 + Kodak24; stored noise-level draw and map per image.
- **Baselines:** blind Restormer/SCUNet-style model, blind DnCNN where available, Neighbor2Neighbor, Noise2Self, and DRUNet with the oracle noise map clearly labeled as an upper-information baseline.
- **Quality model:** the best non-exploited candidate from Setup B, preferably combined with patchwise clean-feature statistics.
- **Objective:** naturalness, residual whiteness, conservative change constraint; no true \(\sigma\) term in the blind track.
- **Evaluation:** standard metrics stratified by noise level and image region, plus local parameter/noise-map correlation.
- **Scientific question:** does local differentiable optimization infer where and how strongly to denoise, or did it only exploit a known global noise level?

This is more aligned with the eventual project, but less diagnostic and less standardized than Setup A.

### Setup D — Real sRGB external-validity test

- **Task:** real smartphone/consumer-camera sRGB denoising, unknown noise.
- **Prior/training data:** no paired images for the proposed optimizer; optionally fit the clean prior on DIV2K. Do not fit on SIDD test or DND.
- **Validation/test:** SIDD validation / SIDD benchmark and DND server.
- **Baselines:** BM3D, Restormer or NAFNet-style supervised model, AP-BSN and one newer self-supervised blind-spot method; report whether each used SIDD pairs, noisy-only SIDD, or external data.
- **Quality model:** frozen NR-IQA ensemble calibrated for smartphone/noise sensitivity, or a clean feature prior; diffusion prior only as a separately costed experiment.
- **Objective:** naturalness + conservative observation/content consistency; no Gaussian residual claim.
- **Evaluation:** official PSNR/SSIM, held-out IQA models, runtime, and qualitative crops. DND is test-only.
- **Scientific question:** does the effect survive correlated camera/ISP noise and domain shift after it has been established under controlled AWGN?

## Recommended commitment now

Commit to Setups A and B as the first research milestone. They share data and outputs, are small enough to run repeatedly, and together separate three causes of failure:

1. **representation failure:** clean-target oracle is weak;
2. **objective failure:** oracle is strong but naturalness-guided optimization is weak;
3. **metric exploitation:** optimized score rises but held-out metrics and fidelity fall.

Proceed to Setup C only if at least one naturalness-guided variant improves the fixed mechanism consistently across both test sets and multiple seeds. Proceed to SIDD/DND only after that result survives blind/noise-varying tests.

This sequence creates a credible environment for deciding whether the central idea has research potential without prematurely attributing failures to satellite imagery, complex camera noise, or an oversized learned model.
