# Research-State Report

## Executive conclusion

The supplied work establishes a coherent research trajectory, but it does **not** establish a competitive denoising result. The project began by adapting the deterministic feature-fusion front end of Self-FuseNet to denoising, then replaced nondifferentiable BRISQUE/NIQE searches with a frozen CycleGAN clean-domain discriminator, and finally explored global weights, local patch weights, alternative losses, trainable filters, and oriented derivative banks.

Three findings survive the audit:

1. Bilateral preconditioning before edge extraction is a plausible and correctly motivated refinement, though its benefit was never measured rigorously.
2. A frozen clean-domain discriminator can provide gradients to a small restoration model; two real discriminator checkpoints survive. Its calibration as an image-quality objective is unverified.
3. Overlapped local optimization with reflected context and Hann aggregation is a useful implementation mechanism for spatially varying parameters.

The strongest performance conclusions do not survive. No result tables, logs, seeds, or saved Self-Fuse outputs were supplied. Several comparison scripts inherit a decisive coefficient-normalization defect, select checkpoints with the test metric, apply BM3D outside its assumed degradation, or report only no-reference scores despite having clean targets.

This report therefore defines two baselines, not a new architecture or a state-of-the-art claim:

- a numerically stable reconstruction of the paper-inspired deterministic block; and
- a corrected complementary-band, three-parameter differentiable baseline whose initial output is exactly the input.

## Evidence boundaries

The labels below are used deliberately.

- **Verified fact:** directly present in the supplied thesis, notes, code, figures, or checkpoint container.
- **Recorded claim:** asserted in the thesis or code comments without recoverable supporting results.
- **Visual observation:** visible in a thesis figure, without a controlled quantitative test.
- **Interpretation:** a conclusion of the mathematical or implementation audit.
- **Hypothesis:** a proposition still needing an experiment.

No old console output, CSV, JSON result, training log, TensorBoard log, Self-Fuse output, Lenna source, or satellite sample was recoverable. The two discriminator files are valid PyTorch checkpoint containers, but their existence proves neither training quality nor usefulness as a denoising prior.

## 1. What the original method was

### The published Self-FuseNet

**Verified fact.** Self-FuseNet is a single-image, data-free remote-sensing **super-resolution** method. It first forms low-, mid-, and high-frequency-derived features with bilateral, Gabor, and derivative operations, searches for perceptually favorable components with BRISQUE, NIQE, and VIF, and fuses them. A customized U-Net and an upsampling convolution then learn from the same image. It is not merely the three deterministic filters, and it is not originally presented as a pure denoiser.

The project implemented and adapted the deterministic self-fusion front end. It did not reproduce the complete customized U-Net/upscaling stage. Consequently, the historical code should be called “Self-FuseNet-inspired denoising,” not a reproduction of Self-FuseNet.

### The project's first denoising pipeline

For an observed image \(y\), the intended components were:

1. **Low-frequency component:** bilateral filtering

   \[
   L(p)=\frac{1}{W_p}\sum_{q\in\Omega_p}
   \exp\!\left(-\frac{\|p-q\|^2}{2\sigma_s^2}
               -\frac{\|y(p)-y(q)\|^2}{2\sigma_r^2}\right)y(q).
   \]

   Spatial proximity and intensity similarity jointly suppress fluctuations while reducing smoothing across strong edges.

2. **High-frequency/edge component:** Sobel or derivative responses, thresholding, two enhancement passes, VIF gating, and sharpening. Later code computes gradients after bilateral preconditioning to reduce noise-edge amplification.

3. **Mid-frequency/texture component:** a bank of oriented Gabor filters,

   \[
   g_{\theta}(u,v)=
   \exp\!\left(-\frac{u_\theta^2+\gamma^2v_\theta^2}{2\sigma^2}\right)
   \cos\!\left(2\pi u_\theta/\lambda+\phi\right),
   \]

   with candidates accepted when BRISQUE and NIQE improved.

4. **Fusion:** a grid over nonnegative coefficients followed by another BRISQUE/NIQE gate. The archived code averages accepted candidates rather than demonstrating that this average is itself optimal.

VIF was used as a full-reference gate between processing stages. BRISQUE and NIQE were used as blind naturalness scores. In denoising without a clean target, however, VIF necessarily treats the noisy observation as its reference. It then measures preservation of information relative to the noisy image, not restoration fidelity to an unknown clean scene.

### Mathematical and practical role of each important component

| Component | Mathematical role | Practical role and interaction |
|---|---|---|
| Gaussian filter | Linear isotropic low-pass convolution | Defines scale spaces and later low/mid/high residual bands; it smooths edges together with noise. |
| Bilateral filter | Nonlinear, data-dependent weighted average in spatial-range space | Supplies an edge-aware smooth component and a less noisy source for Sobel gradients. |
| Sobel/derivative filters | Local first-order directional derivative approximation | Locates strong transitions for selective sharpening; without prefiltering, it responds strongly to noise. |
| Gabor filters | Localized, oriented band-pass correlation | Attempts to represent mid-frequency texture. Signed responses must be preserved; scoring clipped response maps as photographs is not meaningful. |
| Steerable pyramid concept | Multiscale angular analysis with controlled synthesis/reconstruction | Motivates richer directional subbands. The archived implementation never realized this mathematical object. |
| VIF | Full-reference information-fidelity ratio based on a natural-scene/GSM channel model | Gates whether processing preserves information relative to a chosen reference; the noisy-input reference creates a denoising conflict. |
| BRISQUE | Learned no-reference regressor on MSCN and AGGD/GGD-like statistics | Screens images for naturalness without a target. Its score is not a restoration loss or a guarantee of source fidelity. |
| NIQE | Mahalanobis-type distance between test NSS features and a pristine-image model | Provides an opinion-unaware naturalness signal. Domain shift to satellite data and filter maps is a major concern. |
| GGD/AGGD | Parametric models of symmetric/asymmetric normalized coefficient distributions | Underpin BRISQUE/NIQE features; they describe statistics rather than directly removing noise. |
| CycleGAN generators | Unpaired mappings constrained by adversarial, cycle, and identity losses | Produced thesis demonstrations and provided the training context for the discriminator, but also displayed hallucination failures. |
| Multiscale PatchGAN discriminator | Patchwise learned clean-domain score at two image scales | Supplies gradients to restoration parameters; it encodes domain realism, not paired correctness. |
| L1 to noisy input | Pixelwise observation-fidelity regularizer | Prevents large changes, but can also penalize removal of additive noise. |
| SSIM to noisy input | Local structural-statistics fidelity | Preserves local contrast/structure relative to the observation, including some corruption. |
| Total variation | Spatial-gradient sparsity prior | Can suppress fluctuations but oversmooth texture; the archived summed form had resolution-dependent strength. |
| Mean constraint | First-moment preservation | Prevents brightness collapse but does not constrain local content or color sufficiently. |
| Global scalar weights | Very low-dimensional synthesis parameters | Easy to optimize and interpret, but cannot adapt across space or represent complex estimators. |
| Patchwise weights | Piecewise local parameter adaptation | Increases flexibility; overlap-add is needed to prevent seams and must be distinguished from a true dense estimator. |
| PSNR/MSE/MAE | Full-reference error measures | Valid for synthetic paired tests; they measure fidelity but may disagree with perceptual sharpness. |
| SSIM as evaluation | Full-reference structural similarity | Complements error metrics; it must be kept separate from any use as a training/selection loss. |
| BM3D and classical filters | External comparison methods with their own noise assumptions | Useful only when parameters and degradation domains are matched; they are not interchangeable denoising/deblurring baselines. |

## 2. What was modified over time

The project evolved through the following sequence.

1. **Direct deterministic port.** Bilateral, Sobel/derivative, Gabor, VIF, BRISQUE, NIQE, and exhaustive coefficient fusion were adapted from the paper's feature block.
2. **More robust edge extraction.** The paper-style histogram threshold was diagnosed as unstable; Otsu and mean-plus-standard-deviation thresholds were tried, and bilateral preconditioning was adopted.
3. **Differentiable naturalness prior.** A CycleGAN was trained on unpaired noisy and clean domains. The clean-domain PatchGAN discriminator was frozen and reused as a differentiable objective.
4. **Three-scalar per-image optimization.** Low, selected-Gabor mid, and residual high bands were recombined through three optimized coefficients.
5. **Loss variants.** Adversarial loss was combined with input L1, SSIM, TV, mean preservation, and—in a supervised ceiling test—clean-reference losses.
6. **Spatial adaptation.** A whole-image parameter set was replaced first by four nonoverlapping regions and later by dense 64-pixel patches with 50% overlap, reflection padding, and Hann aggregation.
7. **Greater operator freedom.** Sequential operators, unconstrained trainable kernels, random learned edge blocks, and one-scale oriented derivative banks were explored.
8. **Broad comparisons.** Scripts compared the variants with mean, median, Gaussian, bilateral, SVD, nonlocal means, wavelets, total variation, BM3D, and classical unsharp/edge enhancement under several synthetic corruptions.

The complete script-by-script account is in [experiment_ledger.md](experiment_ledger.md).

### Reconstructed data and evaluation setup

The CycleGAN dataset code combines RealBlur, SIDD-small, and a synthetic corruption pool into unpaired clean and degraded domains. Synthetic corruptions include Gaussian and Poisson noise, impulse noise, blur, motion-like blur, compression, and mixtures in different scripts. The thesis records 60,000 noisy and 40,000 clean training images, but the supplied code has no immutable manifest or count record with which to verify those numbers. The dataset's motion-blur construction is not a reliable rotated-line kernel near vertical angles, and its impulse-noise implementation can create channel-colored impulses rather than pixelwise black/white impulses.

Most restoration experiments use one Lenna image, one satellite sample, or crops of those files. Neither image was present in the supplied code archive. Later scripts sometimes use a clean original to synthesize a degradation but still report only BRISQUE. `benchmark.py` is the sole broad script that consistently attempts PSNR, SSIM, and MAE, yet it evaluates one resized Lenna image with one unseeded draw. Therefore, the historical evaluation set is best understood as a collection of development examples, not a benchmark dataset.

## 3. Which modifications actually improved the method

### Verified improvements to correctness or engineering

- **Bilateral-preconditioned Sobel** fixes the clear mechanism by which raw gradients amplify high-frequency noise. This is a better-founded edge operator, although no supplied evidence quantifies its image-quality gain.
- **Frozen discriminator use** makes the naturalness objective differentiable with respect to restoration parameters. It resolves the practical inability of the archived BRISQUE/NIQE implementations to participate in gradient descent.
- **Hann-windowed overlap-add** eliminates the hard seams created by the earlier 2x2 nonoverlapping approach. The cleaned implementation verifies identity reconstruction without seams.
- **Complementary bands and identity-preserving weights** correct the most serious late-stage model defect. At unit weights, the model now returns its input exactly.

### Observed but not quantitatively established

- Thesis figures show that the trained CycleGAN sometimes removes conspicuous noise and that it can also hallucinate or alter content. These are useful qualitative observations, not evidence that its discriminator ranks denoised images correctly.
- The user's reported visual improvement of later local experiments is plausible, particularly because local parameters add spatial flexibility. The output images and controlled comparisons needed to verify it were not supplied.

### Not established

No modification can be said, from the recoverable evidence, to improve average PSNR, SSIM, perceptual quality, or remote-sensing task fidelity over a valid classical baseline. In particular, old “wins” based on BRISQUE alone or on scripts with the coefficient defect are unreliable.

## 4. Invalid or unreliable experiments

The following failure modes recur across the archive.

| Failure | Why it invalidates or weakens the conclusion |
|---|---|
| Three bands sum to \(y\), but their coefficients are normalized to sum to one | Equal initialization produces approximately \(y/3\), not \(y\); many later runs start from a darkened nonidentity image. |
| Test-time checkpoint selected by lowest BRISQUE | The reported metric participates in selection, producing optimistic bias. |
| BRISQUE/NIQE applied to clipped Gabor response maps | These models describe natural-image luminance statistics, not unsigned, clipped filter-response images. |
| BRISQUE and NIQE added as raw numbers | Their scales are not calibrated into a meaningful joint objective. |
| VIF uses the noisy observation as reference | High score can reward retention of the noise that should be removed. |
| Fixed BM3D noise level used for blur, motion blur, Poisson, impulse noise, and clean images | The baseline is deliberately mismatched, and BM3D is not a deblurring algorithm. |
| Only one or two images, unseeded degradations | Results cannot estimate generalization or uncertainty. |
| Clean target exists but only a no-reference metric is reported | Restoration accuracy is left unmeasured despite available ground truth. |
| Nonoverlapping regional processing | Boundary seams confound any benefit from local adaptation. |
| TV is a pixel sum while other losses are means | The effective weight changes with resolution and can dominate unintentionally. |
| Feature matching between unrelated unpaired images | Spatial feature equality has no valid correspondence target. |
| One-scale derivative-of-Gaussian bank called a steerable pyramid | It lacks a multiscale pyramid, synthesis operator, reconstruction guarantee, and correct channel/orientation grouping. |
| Unconstrained “trainable Gabor/Gaussian” kernels | After optimization they are generic convolutions, so conclusions about the named deterministic filters no longer follow. |

These defects do not imply that every underlying hypothesis is bad. The experiment ledger classifies each as a bad formulation, bad implementation of a plausible idea, inconclusive test, or useful retained result.

## 5. Strongest parts of the approach

1. **Data independence at restoration time.** The deterministic path can run on a single image without paired targets.
2. **Interpretability.** Each component has a recognizable spatial/frequency role, and the parameter count can remain very small.
3. **Differentiable parameter adaptation.** A frozen prior can optimize an image-specific mechanism without training that mechanism on paired data.
4. **Local adaptation.** Patchwise parameters can respond to spatially varying texture and noise; overlap-add makes this implementable without obvious block boundaries.
5. **Useful skepticism about blind metrics.** The research correctly recognized that standard BRISQUE/NIQE implementations are awkward optimization objectives and sought a trainable alternative.
6. **Recoverable experimental breadth.** Even though conclusions are missing, the archive identifies the right questions: representational ceiling, prior choice, loss ablation, locality, identity preservation, and comparison with classical methods.

## 6. Fundamental limitations

The core representation has very low capacity. With fixed component maps \(C_k(y)\) and a few scalars,

\[
\hat x=\sum_{k=1}^{K} w_k C_k(y),
\]

the attainable outputs lie in a very small affine or conic set. Spatially local weights enlarge this set, but the mechanism still cannot infer missing detail, express rich signal-dependent shrinkage, distinguish stochastic texture from noise reliably, or model nonlocal/self-similar structure unless those behaviors already exist in the operators.

There is also an identifiability problem. A clean-domain discriminator estimates whether an output resembles its training domain; it does not know which clean image generated \(y\). Without a degradation likelihood or stronger content constraint, many natural-looking outputs may be inconsistent with the observation. The thesis's own CycleGAN failure examples demonstrate this ambiguity.

Finally, the clean prior is only as relevant as its training distribution. Natural photographs, low-light images, and remote-sensing images have different textures, point-spread functions, spectral statistics, compression histories, and semantic structures. A discriminator can reward domain conformity rather than faithful restoration.

## 7. Implementation limitations versus idea limitations

| Implementation choice that can be fixed | Limitation of the underlying idea |
|---|---|
| Identity-breaking coefficient normalization | A handful of global scalars has a low representational ceiling. |
| Unsigned clipping of signed filter responses | Fixed filter banks cannot adapt their basis to every noise/scene combination. |
| Duplicate Gabor orientations and undersized kernels | Blind naturalness alone does not identify the source-consistent clean image. |
| Bad histogram threshold | Using the noisy image as a VIF reference conflicts with removing its noise. |
| Unseeded, one-image benchmarks | Patch-local processing trades flexibility for possible inconsistency and prior exploitation. |
| Untuned or task-mismatched baselines | A learned clean prior can be biased by its domain and training artifacts. |
| No validation split or discriminator calibration | No-reference metrics are imperfect proxies for restoration correctness. |
| Hard seams and incorrect grouped-convolution ordering | Deterministic interpretability alone does not guarantee adequate restoration capacity. |

## 8. Assumptions to keep

These assumptions remain productive working hypotheses, not settled truths.

- Paired clean/noisy targets should not be required by the restoration mechanism.
- Image-specific optimization is worth studying, particularly for domain-specific images.
- Explicit operators and decompositions can provide interpretability and useful inductive bias.
- A differentiable prior may guide low-dimensional or structured parameters.
- Local image statistics matter, so spatial adaptation should remain available.
- Identity preservation and observation consistency should be explicit tests, not hoped-for behavior.
- Classical and modern baselines must be compared under matched degradations and tuning budgets.

## 9. Assumptions to challenge

- “Low, mid, and high frequency” need not mean one bilateral image, one Gabor aggregate, and one Sobel map.
- A weighted sum need not be the correct synthesis rule.
- Better BRISQUE or NIQE need not mean better denoising, especially on remote-sensing imagery.
- A discriminator trained inside a CycleGAN need not be a calibrated image-quality metric.
- Preservation of the noisy input according to VIF need not preserve the desired signal.
- Scalar or patch-constant parameters need not be sufficient.
- Filters called Gaussian, Gabor, or steerable should not be assumed to retain those meanings after unconstrained learning.
- “Data-free at inference” and “no paired targets” do not imply “no learned external prior”; these settings must be named separately.

## 10. Previous ideas worth developing further

This section identifies unfinished questions from the existing work; it does not propose a new architecture.

1. **Representational-ceiling test.** Optimize the current mechanism with clean targets on a held-out image set. Failure under this oracle setting would show that no unsupervised objective can rescue the representation.
2. **Prior calibration.** Test whether the recovered discriminator monotonically ranks controlled degradations, denoising strength, oversmoothing, sharpening, contrast shifts, and hallucinated texture on held-out domains.
3. **Local versus global ablation.** Compare one global parameter set, hard regions, and dense overlap-add using identical objectives and parameter budgets.
4. **Observation-consistency ablation.** Separate the effects of adversarial, L1-to-input, mean, TV, and any degradation-aware constraint with correctly normalized losses.
5. **Verified multiscale oriented analysis/synthesis.** The steerable-pyramid motivation is worth preserving, but the existing implementation must not be reused as evidence.
6. **Identity and clean-image stress tests.** Every method should demonstrate that it does not degrade images that need no restoration.

## Simplified mathematical baseline

### A. Stable paper-inspired deterministic baseline

Let \(y\in[0,1]^{H\times W\times C}\). Define

\[
B=\operatorname{Bilateral}(y),\qquad
M=\mathbf 1\{\|\nabla B\|>\mu_{\nabla B}+\tau\sigma_{\nabla B}\},
\]

\[
E=\operatorname{clip}(y+\eta M\odot \|\nabla B\|,0,1),
\]

and a normalized signed-Gabor energy

\[
T(p)=\operatorname{normalize}\left(
\sqrt{\frac{1}{J}\sum_{j=1}^{J}(g_j*y)(p)^2}
\right).
\]

The reconstructed historical baseline is

\[
\hat x_{\text{paper}}=
\operatorname{clip}\left(
\frac{w_B B+w_E E+w_T T}{w_B+w_E+w_T},0,1
\right),\quad w_k\ge 0.
\]

Unlike the archive, the implementation keeps signed Gabor responses in floating point, removes periodic duplicate orientations, and selects one candidate for a supplied objective rather than averaging all candidates that pass two unrelated gates.

### B. Corrected best late-stage baseline

Let \(G_{\sigma}*y\) denote Gaussian convolution, with \(0<\sigma_f<\sigma_c\). Define complementary bands

\[
L=G_{\sigma_c}*y,\qquad
M=G_{\sigma_f}*y-G_{\sigma_c}*y,\qquad
H=y-G_{\sigma_f}*y.
\]

Then

\[
L+M+H=y
\]

up to numerical boundary handling. The restoration is

\[
\hat x(w)=w_L L+w_M M+w_H H,
\]

with the identity initialization \(w=(1,1,1)\). In the differentiable implementation,

\[
w=3\operatorname{softmax}(a),
\]

so zero logits produce the identity while weights remain positive.

For a frozen multiscale discriminator \(D_s\), the strongest recoverable late-stage objective was

\[
\mathcal L(a)=
\sum_s\|D_s(2\hat x(a)-1)-1\|_2^2
+\lambda_1\|\hat x(a)-y\|_1
+\lambda_\mu|\operatorname{mean}(\hat x(a))-\operatorname{mean}(y)|.
\]

The latter two terms preserve the observation and mean; they are not a physical noise likelihood. For local processing, patch estimates \(\hat x_t\) are synthesized as

\[
\hat x(p)=
\frac{\sum_t h_t(p)\hat x_t(p)}{\sum_t h_t(p)},
\]

where \(h_t\) is a shifted Hann window and reflected padding provides boundary context.

This corrected formulation is the precise baseline for future work. It has not yet been shown to denoise competitively.

## Required evidence before moving to a new mechanism

The reconstruction is complete enough to move forward, but comparisons should use this minimum protocol:

- immutable image manifests and train/validation/test separation where a learned prior is involved;
- fixed random seeds and stored degradation parameters;
- several images and repeated noise draws, with per-image rows rather than only averages;
- full-reference PSNR, SSIM, and error metrics for synthetic pairs, plus carefully labeled no-reference scores for real data;
- noise/task-matched classical baselines;
- hyperparameters selected without consulting the test metric;
- runtime, parameter count, and optimization-step reporting;
- identity, oversmoothing, edge, texture, contrast, and spatially varying-noise stress tests;
- saved configurations, outputs, and machine-readable result tables.

Until those conditions are met, visual quality can motivate a hypothesis but cannot establish improvement.
