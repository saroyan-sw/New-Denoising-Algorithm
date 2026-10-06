# Image restoration that preserves evidence

## Literature review and research proposal

**Research date: 20 September 2026.** Focus: denoising, deblurring, super-resolution, inverse problems, and learned guidance of interpretable restoration operators.

This is a targeted, broad literature review based on primary papers, proceedings, author pages, and implementation repositories. It combines foundational work with papers published through 2026. It is not an exhaustive systematic review or a reproduction of the reported experiments. The 30 core papers below were selected for conceptual relevance, evidence preservation, and usefulness to this project. Additional references cover IQA models and essential background. Code availability means an accessible release or linked implementation; it does not mean that I installed or reproduced it.

## Contents

1. Executive assessment
2. What can “without hallucinations” actually mean?
3. Taxonomy and comparison of methodological families
4. Thirty core papers
5. Differentiable quality and naturalness models
6. Failure modes and negative evidence
7. Closest precedents and novelty assessment
8. Stronger mathematical formulations
9. Experimental plan for this project
10. Five research directions and reading order

## 1. Executive assessment

**The general idea is already substantially explored.** Learning a prior, using it in an optimization objective, and keeping a classical or interpretable restoration mechanism has several established precedents. Automatic denoiser tuning is older still. Relevant routes include SURE-based parameter selection, no-reference content measures, learned variational regularizers, differentiable surrogates of BM3D and camera pipelines, and quality-guided image enhancement. See core papers 1–8 and 30.

**The most promising contribution would concern control and validation:** restrict the allowed edits, use a statistically justified fidelity criterion, make the learned quality model a secondary selector, and demonstrate fewer unsupported structures under metric exploitation and degradation mismatch. Merely combining a filter bank with NIQE, a discriminator, or CLIP-IQA is unlikely to establish a strong methodological contribution.

**An explicit data term does not establish absence of hallucinations.** It controls agreement with measurements. Missing and weakly observed information can still be filled incorrectly, even when the reconstructed image perfectly reproduces the measurement. Measurement-space and null-space hallucinations have already been distinguished explicitly in reconstruction research. [Bhadra et al., 2021](https://arxiv.org/abs/2012.00646).

**A high-quality image is not necessarily the correct image.** Perceptual distribution matching, reconstruction error, and robustness can conflict. Their relationship depends on the degradation, estimator class, and definition of perception. [Blau and Michaeli, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html), [Ohayon et al., ICML 2024](https://proceedings.mlr.press/v235/ohayon24a.html).

**Quality prediction and useful optimization gradients are separate capabilities.** There is direct experimental evidence that the ranking of IQA models changes when they become optimization losses. Modern no-reference models also remain vulnerable to adversarial score manipulation. [Ding et al., IJCV 2021](https://arxiv.org/abs/2005.01338), [Gushchin et al., ICML 2025](https://proceedings.mlr.press/v267/gushchin25a.html).

**My recommendation:** start with AWGN denoising and a small constrained operator family; establish SURE or self-supervised parameter selection as the fidelity baseline; then test whether a learned score improves perception among candidates with similar estimated reconstruction risk. Begin with a modest NR-IQA model and an explicit learned energy as separate alternatives. Treat diffusion guidance as a comparison and later extension.

## 2. What can “without hallucinations” actually mean?

### 2.1 Three different objectives

Assume an unknown image \(x\), observation \(y\), forward operator \(H\), and noise \(n\):

\[
y=Hx+n.
\]

| Goal | Mathematical interpretation | What it establishes | Main limitation |
|---|---|---|---|
| Most likely clean image | MAP: \(\arg\max_x p(x\mid y)\) | A posterior mode under the chosen model | The most probable mode can still be wrong; a prior can dominate ambiguity |
| Lowest expected squared error | MMSE: \(\mathbb E[x\mid y]\) | Minimum expected MSE under the model | Averages incompatible possibilities and can blur uncertain structures |
| Perceptually plausible image | Match natural-image statistics, or draw \(x\sim p(x\mid y)\) | Realism or a possible explanation | A plausible sample need not equal the scene that produced this observation |
| Recover only supported information | Estimate identifiable features; bound uncertainty; abstain from unsupported detail | A qualified evidence claim under stated assumptions | Often cannot provide a unique sharp image everywhere |

MAP and MMSE are different estimators. A stochastic posterior method is not automatically more faithful than a deterministic estimator, and a deterministic method is not automatically conservative.

Also distinguish **marginal realism** \(p_{\hat X}\approx p_X\) from **joint fidelity** \(p_{\hat X,Y}\approx p_{X,Y}\). An unrelated natural photograph can have excellent realism while explaining none of the particular observation. Even an accurate conditional posterior can contain several incompatible images; drawing one does not establish which one occurred.

For this project, the fourth row is the operational version of the user's category 3. It calls for a conservative image **plus information about what remains uncertain**, especially for substantial blur, missing pixels, or high upscaling factors.

### 2.2 Why perfect data consistency is insufficient

For noiseless, consistent linear measurements and the Moore–Penrose pseudoinverse,

\[
\hat x=H^\dagger y+(I-H^\dagger H)z
\]

satisfies \(H\hat x=y\) for any \(z\). The second term lies in the null space. A powerful prior may place convincing but incorrect content there. DDNM operationalizes this decomposition; it does not turn null-space content into observed evidence. [DDNM, ICLR 2023](https://arxiv.org/abs/2212.00490).

Examples:

- Inpainting cannot determine an entirely occluded object from the observed pixels alone.
- Downsampling allows many different high-resolution textures to yield the same low-resolution image.
- Strong blur makes some frequency components so weak that their inversion mostly amplifies noise.
- Even denoising with \(H=I\) is uncertain because the unknown noise realization can explain many candidate clean images. Enforcing \(\hat x=y\) exactly would retain all the noise.

Thus, “the restored image degrades back to the input” is a useful test, but not a proof that its details existed.

### 2.3 A hierarchy of claims

| Claim | What must be checked | What it does not imply |
|---|---|---|
| Algebraic consistency | Correct forward operator and a numerical residual bound | Correct unobserved details |
| Statistical measurement feasibility | Correct likelihood, noise parameters, and operator uncertainty | A unique true image |
| Algorithmic convergence | Stated assumptions on objective, denoiser, and steps | Convergence to the physical truth |
| Calibrated prediction risk | Exchangeability and a valid calibration procedure | Every pixel of every image is correct |
| Semantic correctness | Relevant ground truth, independent measurements, or validated task evidence | Cannot generally be obtained from a naturalness score |

These claims should be reported separately. “Hallucination controlled” should always name the controlled quantity: false edges, measurement inconsistency, feature error, instability, or calibrated miscoverage.

### 2.4 Likelihoods and confidence sets

For known Gaussian noise covariance \(\Sigma\),

\[
L_{\mathrm{meas}}(x;y)=\tfrac12(Hx-y)^T\Sigma^{-1}(Hx-y).
\]

One possible confidence set is

\[
\mathcal C_\alpha(y)=\left\{x:\|\Sigma^{-1/2}(Hx-y)\|_2^2
\leq \chi^2_{m,1-\alpha}\right\},
\]

where \(m\) is the number of independent whitened measurement components. Under the stated model, the true image belongs to this set with probability \(1-\alpha\). **That coverage statement does not certify any image selected from the set.** Estimated noise parameters, clipping, dependent noise, or a wrong \(H\) require a revised analysis.

A lower bound forcing a nonzero residual is not a universal solution: a residual includes removed image content as well as noise, and depends on the estimator. Likewise, forcing its energy to equal \(m\sigma^2\) does not establish that the residual contains only noise.

For photon-limited data, use a Poisson or Poisson–Gaussian likelihood where appropriate. For processed camera RGB, account for the ISP, demosaicing, compression, and potentially correlated noise. A calibrated model for synthetic AWGN should not be presented as a guarantee for arbitrary camera images.

### 2.5 What uncertainty can and cannot do

Posterior samples can reveal alternative plausible structures. Their agreement can be misleading if all samples share a biased prior or an incorrect likelihood. Ensemble variance is therefore a diagnostic, not automatic calibration.

Calibration with representative ground truth can support more precise risk statements. Image-to-image risk-controlling prediction sets demonstrate this route, although their guarantee concerns a specified population risk, not universal correctness. [Angelopoulos et al., ICML 2022](https://proceedings.mlr.press/v162/angelopoulos22a.html).

An evidence-oriented output could contain: a conservative estimate, a map of uncertain features, and separate optional hypotheses. Only the first should be presented as the primary reconstruction.

## 3. Taxonomy and comparison of methodological families

The families overlap. “Test-time optimization” describes when optimization happens; “energy prior” describes the objective; “deterministic” describes the mapping or execution. A method can belong to all three.

| Family and representative work | Intuition / typical objective | Supervision or prior | Evidence preservation | Perception–fidelity trade-off |
|---|---|---|---|---|
| Classical regularization: TV, Tikhonov, sparsity | \(\min_x L_{meas}+\lambda\phi(x)\) | Smoothness, sparsity, bounded variation | Explicit forward model; sometimes convexity and stability | Strong regularization removes real texture or creates staircasing |
| Handcrafted / nonlocal priors: BM3D | Group similar patches; shrink transform coefficients | Repetition and transform sparsity | Local evidence and noise model guide shrinkage | Can remove rare details or mismatch patches |
| Risk-based parameter selection: SURE-LET, Noise2Self | Minimize estimated clean-image risk without clean test targets | Known noise statistics or independent masking | Directly targets reconstruction risk under assumptions | A useful foundation for conservative tuning; not a semantic certificate |
| Learned patch density: EPLL | \(L_{meas}-\lambda\sum_p\log p(P_px)\) | Clean-patch probability model | Explicit likelihood remains separate | Generic patch statistics can replace unusual structures |
| Learned energy / discriminator: adversarial regularizers, WCRR | \(L_{meas}+\lambda E_\phi(x)\) | Clean corpus, sometimes corrupted examples or supervised pairs | Explicit energy; architectural restrictions can support convergence | Energy favors its training distribution |
| Interpretable processing with learned control: Exposure, ProxyOpt | Optimize or predict parameters of \(R(y;\theta)\) | Quality, targets, adversarial or task signals | Reachable outputs restricted by operators | Control model can still choose destructive settings |
| Implicit architectural prior: DIP | \(\min_\theta\|Hf_\theta(z)-y\|^2\) | Untrained CNN and optimization trajectory | Data fitting and early stopping | Late iterations fit noise; architecture supplies unsupported preferences |
| Noisy / self-supervised learning: N2N, N2Void, N2Self | Noisy targets or masked prediction | Independence / unbiasedness assumptions | Avoids requiring clean training targets | Correlated noise, masking leakage, and model bias remain |
| Plug-and-play: PnP, DPIR, Prox-PnP | Alternate data proximal step with a denoiser | Handcrafted or pretrained denoiser | Explicit measurement step; some convergence results | Fixed point need not correspond to true image or explicit density |
| RED and score regularization: RED, SNORE | Use \(x-D(x)\) or a smoothed score | Denoiser learned from clean images and synthetic noise | Fidelity plus denoiser-based update | Energy interpretation needs assumptions; out-of-distribution updates matter |
| Diffusion prior optimization: Graikos, RED-diff | Optimize image using a frozen multiscale denoising prior | Pretrained diffusion model | Likelihood plus prior objective | Flexible prior can choose a wrong plausible explanation |
| Constrained generative restoration: DDNM, DiffPIR | Projection or data solve interleaved with generative updates | Pretrained generative prior | Stronger control of observable components | Null-space detail can still be invented |
| Approximate posterior sampling: DPS, DAPS, SITCOM | Approximate \(p(x\mid y)\), often with likelihood guidance | Learned score and measurement model | Incorporates data; some algorithms improve sampling consistency | Approximation bias and sample ambiguity remain |
| NR-IQA / preference guidance: LPIE, TTPO, RAR | Maximize score or preference with other constraints | Human quality labels / foundation models | Depends entirely on separate content and measurement controls | Score exploitation and semantic drift |
| Feature / perceptual priors: LPIPS, DISTS, CLIP, DINO | Compare features or score alignment | Recognition / human judgments / text-image data | Can penalize some content changes | Feature invariance discards exact spatial and texture evidence |
| Frequency-restricted pipelines | Bounded gains or shrinkage per band | Noise spectrum, operator transfer function | Makes allowable changes inspectable | Ringing, lost phase information, or unstable high-frequency gain |
| Calibrated conservative restoration | Estimate image plus risk sets / abstention | Calibration data and an explicit risk definition | Quantified error control under assumptions | May retain uncertainty or sacrifice apparent sharpness |

Foundational links: [PnP](https://brendt.wohlberg.net/publications/venkatakrishnan-2013-plugandplay.html), [RED](https://arxiv.org/abs/1611.02862), [RED clarification](https://arxiv.org/abs/1806.02296), [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html). The specific modern methods are documented in Section 4.

### Important mathematical distinctions

**PnP is not automatically MAP.** Replacing a proximal operator with a generic denoiser can produce a useful equilibrium without an associated explicit regularizer.

**RED needs care.** For \(\rho(x)=\frac12x^T(x-D(x))\), the commonly used gradient \(x-D(x)\) requires conditions including local homogeneity and a symmetric denoiser Jacobian. General neural denoisers do not satisfy these automatically. [Reehorst and Schniter, IEEE TCI 2019](https://arxiv.org/abs/1806.02296).

**Scores are gradients of smoothed densities.** For an ideal Gaussian denoiser,

\[
D_\sigma(z)-z=\sigma^2\nabla_z\log p_\sigma(z).
\]

This relates a denoiser to a score of the noisy-image distribution. Applying a pretrained denoiser to arbitrary optimization iterates can still create a distribution mismatch; stochastic re-noising is one studied remedy. [SNORE](https://proceedings.mlr.press/v235/renaud24a.html).

**A discriminator is not automatically an image likelihood.** For an ideal balanced binary discriminator,

\[
\operatorname{logit}D^*(x)=\log p_{clean}(x)-\log p_{negative}(x).
\]

It learns a density ratio against its negative distribution. A CycleGAN discriminator is tied to the corresponding generator and training domain; it may ignore new artifacts. A Wasserstein critic has a different mathematical role again. An explicit learned energy is often easier to formulate as a regularizer than an arbitrary “realness” score. [Adversarial regularizers](https://papers.nips.cc/paper_files/paper/2018/hash/d903e9608cfbf08910611e4346a0ba44-Abstract.html).

**Bilevel learning and test-time tuning are different.** One can learn a global prior or controller through an inner restoration solver, then freeze it. Alternatively, one can optimize a small \(\theta\) separately for each observation. Either can use interpretable operators; neither by itself proves fidelity.

## 4. Thirty core papers

Each entry states the task, supervision, prior, optimization objective, fidelity mechanism, limitations, evaluation, and release status. Dataset names are representative rather than an exhaustive inventory. “No paired targets” does not mean “no clean images were used”: pretrained priors often require substantial clean-image training. Reported metrics below identify what the authors measured; they are not a cross-paper leaderboard.

### A. Closest predecessors to learned guidance of deterministic operators

### 1. Blu and Luisier — The SURE-LET Approach to Image Denoising

**IEEE TIP, 2007.** [Paper](https://bigwww.epfl.ch/preprints/blu0701p.pdf).

- **Task / supervision:** Gaussian denoising without clean training or test targets; known noise variance.
- **Prior / objective:** Express the denoiser as a linear expansion of elementary functions and choose coefficients by minimizing Stein's unbiased risk estimate, yielding tractable coefficient optimization.
- **Control / limits:** Targets expected squared reconstruction error under the noise assumptions. Does not certify individual edges; model mismatch and adaptive selection matter. Originally formulated with transform-domain estimators, with a broader connection to combining denoisers.
- **Evidence / release:** Standard grayscale test images; PSNR and computational cost. [Author resources and related SURE implementations](https://bigwww.epfl.ch/research/projects/denoising.html).

**Relevance:** a necessary predecessor and baseline for our weighted filter ensemble.

### 2. Zhu and Milanfar — Automatic Parameter Selection for Denoising Algorithms Using a No-Reference Measure of Image Content

**IEEE TIP, 2010.** [Author paper page](https://research.google/pubs/automatic-parameter-selection-for-denoising-algorithms-using-a-no-reference-measure-of-image-content/).

- **Task / supervision:** Tune denoising parameters from the noisy image, without a clean reference.
- **Prior / objective:** A handcrafted content measure based on singular values of local gradient matrices balances structure and noise; select operator parameters using this score.
- **Control / limits:** Low-dimensional algorithm tuning limits possible outputs, but the score is not a measurement likelihood or universal fidelity guarantee. Content and degradation assumptions affect selection.
- **Evidence / release:** Synthetic and real noisy natural images; reconstruction-error evaluation and subjective-quality correlations, including TID2008. Complete author code release not verified.

**Relevance:** directly anticipates the high-level idea of a no-reference score selecting denoiser parameters.

### 3. Zoran and Weiss — From Learning Models of Natural Image Patches to Whole Image Restoration

**ICCV, 2011; EPLL.** [Paper](https://people.csail.mit.edu/danielzoran/EPLLICCVCameraReady.pdf).

- **Task / supervision:** Denoising, deblurring, inpainting; clean patch corpus for prior learning, no paired inverse-problem training requirement.
- **Prior / objective:** Gaussian-mixture patch density; minimize measurement error minus expected patch log likelihood through half-quadratic splitting.
- **Control / limits:** Explicit likelihood and overlapping patch agreement. Small patches capture limited global structure; unusual image content can be pulled toward learned patch statistics.
- **Evidence / release:** Berkeley natural-image tests including 68-image denoising evaluation; PSNR and runtime. [Author page with paper and software resources](https://people.csail.mit.edu/danielzoran/); historical MATLAB implementation, environment compatibility untested.

**Relevance:** a clean example of learned naturalness as an explicit optimization regularizer.

### 4. Lunz, Öktem, and Schönlieb — Adversarial Regularizers in Inverse Problems

**NeurIPS, 2018.** [Paper](https://papers.nips.cc/paper_files/paper/2018/hash/d903e9608cfbf08910611e4346a0ba44-Abstract.html).

- **Task / supervision:** Denoising and CT; unpaired clean examples and unregularized reconstructions train a Wasserstein-style critic.
- **Prior / objective:** Freeze the learned scalar regularizer and optimize data fit plus weighted network energy.
- **Control / limits:** Explicit measurement model; regularization analysis under assumptions. Nonconvex optimization and critic distribution mismatch remain; this is not a hallucination-free theorem.
- **Evidence / release:** BSDS500 natural-image crops and LIDC/IDRI CT; PSNR, SSIM, and visual reconstruction comparisons. [Official code](https://github.com/lunz-s/DeepAdverserialRegulariser); pretrained-package completeness unverified.

**Relevance:** the strongest foundational match to “use a learned discriminator-like prior inside a classical inverse solver.”

### 5. Talebi and Milanfar — Learned Perceptual Image Enhancement

**ICCP, 2018.** [Paper](https://arxiv.org/abs/1712.02864).

- **Task / supervision:** Tone mapping and dehazing. MIT–Adobe FiveK images processed by existing operators provide reference outputs; NIMA uses prior quality/aesthetic training.
- **Prior / objective:** Train a context aggregation network with reference-output MSE plus a frozen NIMA quality reward.
- **Control / limits:** The MSE term anchors enhancement; large quality-loss weights can cause distortions. Crucially, this trains an operator-approximating network, rather than only tuning classical filter knobs at test time.
- **Evidence / release:** FiveK enhancement examples and NIMA-based quantitative/visual evaluation. NIMA implementations exist; complete original enhancement release not verified.

**Relevance:** a direct precedent for quality regularization, with an explicit warning about its weight.

### 6. Hu et al. — Exposure: A White-Box Photo Post-Processing Framework

**ACM TOG / SIGGRAPH, 2018.** [Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/01/Exposure.pdf).

- **Task / supervision:** Photo enhancement from unpaired examples of desired appearance; no aligned clean restoration targets.
- **Prior / objective:** A learned controller selects differentiable, interpretable editing operations using adversarial learning and reinforcement learning.
- **Control / limits:** Restricted operations make edits understandable. Aesthetic/style matching does not establish measurement fidelity; aggressive tone and color changes remain possible.
- **Evidence / release:** MIT–Adobe FiveK and style-oriented image collections; user judgments and appearance-distribution comparisons. [Code and pretrained models](https://github.com/yuanming-hu/exposure).

**Relevance:** learned guidance of an interpretable processing pipeline is established, even though its goal differs from evidence-preserving denoising.

### 7. Tseng et al. — Hyperparameter Optimization in Black-Box Image Processing Using Differentiable Proxies

**ACM TOG / SIGGRAPH, 2019; ProxyOpt.** [Project and paper](https://www.cs.princeton.edu/~fheide/proxyopt).

- **Task / supervision:** Tune BM3D and a camera ISP. A learned differentiable proxy approximates the black box; denoising experiments use clean targets.
- **Prior / objective:** Backpropagate reconstruction or downstream-task losses through the proxy to processing parameters, then apply the real pipeline.
- **Control / limits:** Parameters remain interpretable, but proxy error can mislead optimization. Task-optimal output may look unnatural; a clean-reference objective is not reference-free test-time naturalness tuning.
- **Evidence / release:** SIDD denoising and camera/object-detection experiments; PSNR and task metrics. Complete official code release not verified on the project page.

**Relevance:** the most direct technical predecessor for differentiating through classical restoration parameter choices.

### 8. Khondaker and Ray — Learning Instance-Specific Parameters of Black-Box Models Using Differentiable Surrogates

**WACV, 2025.** [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Khondaker_Learning_Instance-Specific_Parameters_of_Black-Box_Models_using_Differentiable_Surrogates_WACV_2025_paper.pdf).

- **Task / supervision:** BM3D denoising with supervised reconstruction examples.
- **Prior / objective:** Learn a BM3D surrogate and a second network predicting image-dependent parameters; optimize reconstruction quality through the surrogate.
- **Control / limits:** Final processing uses BM3D. Learned parameter prediction and surrogate accuracy determine generalization; higher noise makes tuning harder. This is not a frozen IQA prior with only per-image test-time optimization.
- **Evidence / release:** SIDD and CBSD68; PSNR and SSIM. [Official code](https://github.com/arnisha-k/instance-specific-param); exact pretrained deployment package not independently tested.

**Relevance:** recent close prior art that must be compared before claiming novelty for image-specific classical-filter tuning.

### B. Self-supervision, deep priors, and explicit regularization

### 9. Ulyanov, Vedaldi, and Lempitsky — Deep Image Prior

**CVPR, 2018.** [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html).

- **Task / supervision:** Denoising, super-resolution, inpainting; no external clean training set.
- **Prior / objective:** An untrained convolutional network maps fixed noise to an image; optimize its weights against the degraded observation.
- **Control / limits:** Forward-model loss and early stopping. The architecture and optimization trajectory impose an implicit prior; eventual noise fitting and unsupported inpainting remain possible.
- **Evidence / release:** Standard natural-image restoration examples and SR benchmarks including Set5/Set14; PSNR and visual comparisons. [Official notebooks](https://github.com/DmitryUlyanov/deep-image-prior); no pretrained model needed.

**Relevance:** establishes test-time optimization without learning a restoration mapping across a dataset, but uses many less interpretable parameters.

### 10. Batson and Royer — Noise2Self: Blind Denoising by Self-Supervision

**ICML, 2019.** [Paper](https://proceedings.mlr.press/v97/batson19a.html).

- **Task / supervision:** Denoising without clean targets, assuming suitable conditional independence and unbiased noise.
- **Prior / objective:** A J-invariant predictor excludes the held-out noisy component; squared prediction loss differs from clean-image MSE by a noise constant. Supports classical tuning and neural training.
- **Control / limits:** Statistical justification comes from masking and noise assumptions. Correlation or leakage invalidates it; masking can lose useful local information.
- **Evidence / release:** Hànzì, ImageNet, CellNet and biological applications; PSNR and repeated-run comparisons. [Official code](https://github.com/czbiohub-sf/noise2self).

**Relevance:** a principled alternative to naturalness alone when choosing parameters without clean test images.

### 11. Hurault, Leclaire, and Papadakis — Proximal Denoiser for Convergent Plug-and-Play Optimization with Nonconvex Regularization

**ICML, 2022; Prox-PnP.** [Paper](https://proceedings.mlr.press/v162/hurault22a.html).

- **Task / supervision:** Denoising and inverse restoration; denoiser trained with clean/noisy examples.
- **Prior / objective:** Construct a denoiser with a proximal interpretation, then use splitting algorithms with a data term. Training includes denoising loss and Jacobian-related restrictions.
- **Control / limits:** Convergence results require specified smoothness and step conditions. Practical soft enforcement is not a global certification, and convergence does not imply truth.
- **Evidence / release:** CBSD68-derived tests, blur kernels, SR and inpainting; PSNR, objective behavior, and Jacobian diagnostics. [Official code](https://github.com/samuro95/Prox-PnP).

**Relevance:** teaches how to make a learned prior compatible with a controlled optimizer.

### 12. Renaud et al. — Plug-and-Play Image Restoration with Stochastic deNOising REgularization

**ICML, 2024; SNORE.** [Paper](https://proceedings.mlr.press/v235/renaud24a.html).

- **Task / supervision:** Deblurring and inpainting; pretrained denoiser, without paired training for each inverse task.
- **Prior / objective:** Stochastic optimization of fidelity plus a noise-smoothed prior; inject noise before denoising to better match the denoiser's training conditions.
- **Control / limits:** Convergence analysis under assumptions, with approximation considerations. Prior and forward-model mismatch remain.
- **Evidence / release:** CBSD10 restoration tests; PSNR, SSIM, LPIPS, and BRISQUE. [Official code](https://github.com/Marien-RENAUD/SNORE).

**Relevance:** a defensible way to use denoiser-derived prior information. Restricting its updates to our parameter space would be an adaptation, not something this paper already demonstrates.

### 13. Graikos et al. — Diffusion Models as Plug-and-Play Priors

**NeurIPS, 2022.** [Paper](https://papers.nips.cc/paper_files/paper/2022/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html).

- **Task / supervision:** Constrained inference using a pretrained diffusion model; no additional task-specific model training.
- **Prior / objective:** Optimize through the frozen denoising model using a diffusion-based variational objective and differentiable task constraints.
- **Control / limits:** Constraints guide solutions; prior-driven completion remains possible. This is a general inference framework, not a standard natural-photo denoising benchmark.
- **Evidence / release:** MNIST/face examples, segmentation and combinatorial demonstrations; task-specific evaluation and visual results. [Official code](https://github.com/AlexGraikos/diffusion_priors).

**Relevance:** pretrained generative models can serve inside an objective without training a new restoration network.

### 14. Mardani et al. — A Variational Perspective on Solving Inverse Problems with Diffusion Models

**ICLR, 2024; RED-diff.** [Paper](https://arxiv.org/abs/2305.04391).

- **Task / supervision:** Multiple inverse problems using pretrained diffusion priors; no paired training for each restoration task.
- **Prior / objective:** Variational posterior approximation yields a measurement term plus multiscale denoising regularization; optimize image variables using stochastic gradients.
- **Control / limits:** Likelihood-aware, but posterior approximation and learned-prior bias remain. It does not recover a unique true null-space component.
- **Evidence / release:** ImageNet-based inverse-problem evaluation; reported reconstruction and distributional metrics, including PSNR and FID. [Official code and checkpoint links](https://github.com/NVlabs/RED-diff), including ImageNet and FFHQ priors.

**Relevance:** close to “frozen pretrained prior as an optimization guide,” with a much larger image search space.

### C. Measurement-aware generative restoration

### 15. Wang, Yu, and Zhang — Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model

**ICLR, 2023; DDNM.** [Paper](https://arxiv.org/abs/2212.00490).

- **Task / supervision:** Linear inverse problems: SR, inpainting, colorization, deblurring and compressed sensing; pretrained image prior.
- **Prior / objective:** Correct the observable component of diffusion predictions while allowing prior-driven null-space refinement; noisy extension DDNM+.
- **Control / limits:** Exact consistency applies to the ideal noiseless linear construction. Neither exact consistency nor the noisy extension guarantees correct missing content.
- **Evidence / release:** ImageNet and face-image experiments; PSNR, FID, and sampling comparisons. [Official code](https://github.com/wyhuai/DDNM) with pretrained-model instructions.

### 16. Chung et al. — Diffusion Posterior Sampling for General Noisy Inverse Problems

**ICLR, 2023; DPS.** [Paper](https://arxiv.org/abs/2209.14687).

- **Task / supervision:** Noisy linear and nonlinear inverse problems, including Gaussian/Poisson settings; pretrained diffusion prior.
- **Prior / objective:** Approximate likelihood guidance through the predicted clean image during reverse diffusion.
- **Control / limits:** Incorporates observations without strict projection. Approximate gradients, tuning, and multimodal posteriors limit fidelity; “posterior sampling” does not imply exact samples.
- **Evidence / release:** FFHQ and ImageNet; tasks include blur, SR, inpainting and phase-related recovery; PSNR/SSIM, LPIPS, and FID. [Official implementation](https://github.com/DPS2022/diffusion-posterior-sampling).

### 17. Zhu et al. — Denoising Diffusion Models for Plug-and-Play Image Restoration

**CVPR NTIRE workshop, 2023; DiffPIR.** [Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Zhu_Denoising_Diffusion_Models_for_Plug-and-Play_Image_Restoration_CVPRW_2023_paper.html).

- **Task / supervision:** SR, deblurring and inpainting using a pretrained diffusion model.
- **Prior / objective:** Half-quadratic splitting combines data subproblems with diffusion denoising/sampling.
- **Control / limits:** Explicit data updates improve control; finite sampling, prior bias and null-space ambiguity remain. This is a workshop paper, not a main-conference ICCV paper.
- **Evidence / release:** FFHQ/ImageNet restoration; PSNR and perceptual/distributional metrics such as LPIPS and FID; sampling-cost comparisons. [Official code](https://github.com/yuanzhi-zhu/DiffPIR).

### 18. Zhang et al. — Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing

**CVPR, 2025; DAPS** (preprint 2024). [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Improving_Diffusion_Inverse_Problem_Solving_with_Decoupled_Noise_Annealing_CVPR_2025_paper.pdf).

- **Task / supervision:** Linear and nonlinear inverse problems; pretrained diffusion prior.
- **Prior / objective:** Separate denoising, likelihood-based sampling, and re-noising so early trajectory errors can be corrected more substantially.
- **Control / limits:** Better use of measurement likelihood; idealized sampling arguments differ from practical learned models and finite MCMC. Compute budgets matter.
- **Evidence / release:** FFHQ/ImageNet, including SR, blur, phase retrieval and HDR; PSNR, SSIM and LPIPS. [Author paper with code link](https://arxiv.org/html/2407.01521v3).

### 19. Alkhouri et al. — SITCOM: Step-wise Triple-Consistent Diffusion Sampling for Inverse Problems

**ICML, 2025.** [Paper](https://proceedings.mlr.press/v267/alkhouri25b.html).

- **Task / supervision:** Linear and nonlinear restoration with pretrained diffusion models; no paired task-specific retraining.
- **Prior / objective:** Stepwise optimization enforces measurement, backward and forward consistency around diffusion predictions.
- **Control / limits:** Multiple consistency conditions improve the solver; they do not establish semantic correctness or exact posterior calibration.
- **Evidence / release:** FFHQ/ImageNet inverse-problem comparisons and medical reconstruction experiments; PSNR, SSIM, LPIPS and runtime. [Official code](https://github.com/sjames40/SITCOM), with model and data setup instructions.

### D. Limits, uncertainty, and hallucination evaluation

### 20. Blau and Michaeli — The Perception–Distortion Tradeoff

**CVPR, 2018.** [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html).

- **Problem / supervision:** Theory and evaluation of restoration objectives; no new restoration training recipe.
- **Prior / objective:** Relates expected distortion to distributional perceptual quality, defined through distinguishability from real images.
- **Control / limits:** Explains why better realism can cost reconstruction accuracy. Does not say every individual improvement must worsen fidelity, nor that all methods already lie on the optimal boundary.
- **Evidence / release:** Mathematical analysis and 4× SR comparisons on BSD100, using RMSE, SSIM, MS-SSIM, IFC, VIF, VGG features and an NR perception score; restoration code/checkpoints are not the main contribution.

### 21. Ohayon, Michaeli, and Elad — The Perception–Robustness Tradeoff in Deterministic Image Restoration

**ICML, 2024.** [Paper](https://proceedings.mlr.press/v235/ohayon24a.html).

- **Problem / supervision:** Theory for deterministic estimators and ambiguous inverse problems; empirical pretrained-restorer analysis.
- **Prior / objective:** Studies closeness of the reconstruction–measurement joint distribution to the true joint distribution and its relation to estimator sensitivity.
- **Control / limits:** Under the theorem's non-invertibility assumptions, approaching perfect joint perceptual quality imposes severe robustness costs. This is not an unconditional claim that all useful deterministic denoisers are unstable.
- **Evidence / release:** SR and face-restoration experiments, including CelebA-HQ examples; perceptual quality, reconstruction and input-perturbation sensitivity. Complete official implementation release not verified.

### 22. Bhadra et al. — On Hallucinations in Tomographic Image Reconstruction

**IEEE TMI, 2021.** [Paper](https://arxiv.org/abs/2012.00646).

- **Problem / supervision:** Define and expose hallucinations in learned reconstruction; ground truth is used for analysis.
- **Prior / objective:** Decompose reconstruction into measurement and generalized null components; use hallucination maps to study learned-prior effects.
- **Control / limits:** A diagnostic framework, not a guarantee-producing restoration algorithm. Some error components need inaccessible truth for evaluation.
- **Evidence / release:** Simulated reconstruction with adult fastMRI brain images and pediatric epilepsy MRI distribution shift; hallucination/error maps and reconstruction comparisons. [Official code](https://github.com/comp-imaging-sci/hallucinations-tomo-recon).

### 23. Angelopoulos et al. — Image-to-Image Regression with Distribution-Free Uncertainty Quantification and Applications in Imaging

**ICML, 2022.** [Paper](https://proceedings.mlr.press/v162/angelopoulos22a.html).

- **Problem / supervision:** Calibrated prediction sets for imaging; representative reference targets are needed for training/calibration.
- **Prior / objective:** Pixelwise quantile estimates and calibrated set expansion control a chosen prediction risk.
- **Control / limits:** With specified confidence over calibration, expected test-image miscoverage is controlled under exchangeability. This is not conditional coverage for every pixel or simultaneous correctness of an image.
- **Evidence / release:** Quantitative phase microscopy, knee MRI, and Drosophila TEM super-resolution; interval sizes, miscoverage and stratified risk. [Official code](https://github.com/aangelopoulos/im2im-uq).

### 24. Kim et al. — HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration

**CVPR, 2026.** [Paper](https://arxiv.org/abs/2512.03345).

- **Problem / supervision:** Controlled hallucination generation and evaluation; reference images and annotations support benchmarking.
- **Prior / objective:** Diffusion-based perturbation produces targeted, plausible errors; distinguishes measurement-inconsistent from measurement-consistent hallucinations.
- **Control / limits:** Evaluates failures rather than guaranteeing safe restoration. Its SHAFE feature comparison uses a reference; separate learned detectors should not be confused with that reference-based score.
- **Evidence / release:** Brain MRI derived from HCP, with industrial demonstrations; 4,350 annotated hallucination examples, detection/severity and localization evaluation. [Code/data project](https://github.com/edshkim98/HalluGen), marked as active development.

### E. Quality optimization and recent feedback methods

### 25. Ding et al. — Comparison of Full-Reference Image Quality Models for Optimization of Image Processing Systems

**IJCV, 2021.** [Paper](https://arxiv.org/abs/2005.01338).

- **Problem / supervision:** Compare IQA models as losses, using clean references and human evaluation.
- **Prior / objective:** Eleven full-reference objectives guide denoising, deblurring, SR and compression; also examines direct reference recovery.
- **Control / limits:** Finds task-dependent optimization behavior; good quality prediction does not imply good image optimization. Full-reference results do not directly validate NR-IQA guidance.
- **Evidence / release:** Waterloo Exploration source images; subjective studies and objective comparisons. MAE/MS-SSIM were strong for denoising in this protocol, while learned metrics benefited other tasks. [Code](https://github.com/dingkeyan93/IQA-optimization).

### 26. Gushchin et al. — Guardians of Image Quality: Benchmarking Defenses Against Adversarial Attacks on Image Quality Metrics

**ICML, 2025.** [Paper](https://proceedings.mlr.press/v267/gushchin25a.html).

- **Problem / supervision:** Robustness of NR-IQA to score manipulation; uses existing models, attacks, quality labels and human judgments.
- **Objective:** Benchmark defenses under adversarial optimization, including adaptive attacks.
- **Control / limits:** Demonstrates practical vulnerability and imperfect defenses; no universal protection. The final paper evaluates 30 defenses, 14 attacks and nine NR models.
- **Evidence / release:** KonIQ-10k, KADID-10k, NIPS 2017 images and AGIQA-3K; clean/attacked correlations, defense scores, image distortion, runtime and human responses. [Benchmark resources](https://videoprocessing.ai/benchmarks/iqa-defenses.html).

### 27. Li et al. — Test-Time Preference Optimization for Image Restoration

**AAAI, 2026; TTPO.** [Paper](https://arxiv.org/abs/2511.19169).

- **Task / supervision:** Improve existing restoration outputs without retraining the restorer; pretrained diffusion and quality models still contain learned supervision.
- **Prior / objective:** Generate edited candidates, rank preferences, and use preferred/dispreferred examples for diffusion guidance, with initial-image and frequency-related constraints.
- **Control / limits:** Limited edit strength helps preserve content but does not enforce the original physical measurement. The paper shows structural changes at larger noise scales.
- **Evidence / release:** Kodak, BSD100, RealSR, Rain100L and CDD11; NR-IQA and user comparisons. [Repository](https://github.com/lbc12345/TTPO) states original code release is pending; do not treat it as a complete reproducible release.

### 28. Man et al. — ELAD: Blind Face Restoration Using Expectation-Based Likelihood Approximation and Diffusion Prior

**SIGGRAPH Asia Conference Papers, 2025.** [Published project](https://man-sean.github.io/elad-website/).

- **Task / supervision:** Blind face restoration; pretrained face prior and learned degradation estimation, without end-to-end training of a new restoration network.
- **Prior / objective:** Approximate degradation likelihood using its moments, then guide a diffusion prior.
- **Control / limits:** Explicitly investigates distortion and consistency when the degradation is uncertain; learned likelihood errors and face-prior bias remain.
- **Evidence / release:** Synthetic CelebA-based tests and real faces including LFW/WebPhoto/WIDER examples; distortion, perceptual and consistency comparisons. Code advertised as forthcoming on the checked project page.

The related earlier manuscript, [Proxies for Distortion and Consistency with Applications for Real-World Image Restoration](https://arxiv.org/abs/2501.12102), develops evaluation proxies. Treat its title/version separately from the published ELAD paper.

### 29. Chen et al. — Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration

**CVPR, 2026; RAR.** [Project](https://restore-assess-repeat.github.io/).

- **Task / supervision:** Composite and unknown degradations; end-to-end learned restoration and quality assessment using restoration datasets.
- **Prior / objective:** Shared latent assessment and a flow-matching restoration backbone iteratively identify degradation, restore and verify quality.
- **Control / limits:** Quality feedback and stopping improve adaptivity; the system remains generative and lacks a measurement-based hallucination guarantee.
- **Evidence / release:** DIV2K/Flickr2K, SIDD, GoPro, LOL, RESIDE and rain datasets; mixed-degradation MiO100 tests. In composite Group A, RAR has lower LPIPS than AgenticIR (0.1299 versus 0.3148), but lower PSNR (20.46 versus 21.04 dB): a concrete perception/fidelity divergence. [Result table](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.pdf), [official code](https://github.com/saic-fi/RAR).

**Relevance:** assessment-driven restoration is current research, but differs from a frozen scorer tuning a small deterministic pipeline.

### 30. Goujon, Neumayer, and Unser — Learning Weakly Convex Regularizers for Convergent Image-Reconstruction Algorithms

**SIAM Journal on Imaging Sciences, 2024.** [Paper](https://doi.org/10.1137/23M1565243).

- **Task / supervision:** Denoising and CT/MRI reconstruction; supervised denoising trains a compact prior.
- **Prior / objective:** Learn filters and spline nonlinearities through denoising with an L1 training loss; optimize fidelity plus the resulting weakly convex energy. Appropriate denoising settings yield an overall convex objective.
- **Control / limits:** Interpretable parameters and convergence properties; general inverse objectives need not remain convex. These guarantees concern optimization, not absence of hallucination.
- **Evidence / release:** BSD500 natural-image training, fastMRI knee images and low-dose CT Grand Challenge scans; PSNR, SSIM, model size and convergence. [Full manuscript](https://arxiv.org/pdf/2308.10542), [code, trained models and tutorials](https://github.com/axgoujon/weakly_convex_ridge_regularizer).

**Relevance:** particularly strong for an interpretable learned prior with mathematical structure.

### Additional foundations and useful comparators

- **Noise2Noise, ICML 2018:** independent aligned noisy targets can replace clean targets for suitable losses; this requires stronger data access than a single noisy image. [Paper](https://proceedings.mlr.press/v80/lehtinen18a.html).
- **Noise2Void, CVPR 2019:** masked prediction trains denoisers from single noisy observations without clean or repeated noisy targets. Independence and predictable image structure remain assumptions. [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.html).
- **Zero-Shot Noise2Noise, CVPR 2023:** fits a small network to one noisy image using internal constructions; relevant for a compute-limited single-image baseline. Pixelwise independence is central. [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html).
- **Trainable Nonlinear Reaction Diffusion, TPAMI 2017:** learns filters and influence functions in an explicit processing scheme; useful prior art against broad claims about learned interpretable filtering. [Paper](https://arxiv.org/abs/1508.02848).
- **RED, SIAM JIS 2017:** foundational denoiser-based regularization. Read alongside the IEEE TCI 2019 clarification rather than assuming every denoiser defines the original RED energy. [Paper](https://arxiv.org/abs/1611.02862), [code](https://github.com/google/RED).
- **CM4IR, CVPR 2025:** few-step consistency-model restoration is useful for a compute comparison; a “consistency model” is a model family, not by itself a measurement-fidelity certificate. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Garber_Zero-Shot_Image_Restoration_Using_Few-Step_Guidance_of_Consistency_Models_and_CVPR_2025_paper.pdf), [code](https://github.com/tirer-lab/CM4IR).
- **CLIP-LIT, ICCV 2023:** iteratively learned prompts guide unpaired backlit enhancement. Useful foundation-model guidance precedent, but it trains an enhancer and targets exposure/appearance. [Author code and paper links](https://github.com/ZhexinLiang/CLIP-LIT).

Publication distinctions matter: DiffPIR and MANIQA are CVPR workshop papers; DAPS is CVPR 2025 despite its 2024 preprint; DISTS appeared online before its 2022 TPAMI issue. Recent dates alone do not determine suitability for evidence preservation.

## 5. Differentiable quality and naturalness models

### 5.1 The distinction that determines usefulness

We need to distinguish four properties:

1. **Quality prediction:** Does the score correlate with human judgments on a dataset?
2. **Computational differentiation:** Can gradients reach the input tensor through the actual implementation?
3. **Optimization usefulness:** Do steps following those gradients improve human-perceived quality?
4. **Restoration fidelity:** Do the steps preserve the underlying scene?

Success on one property does not establish the others. Reported SRCC/PLCC measure prediction monotonicity/correlation, not the quality of optimized images. A model may distinguish blur from sharpness yet reward excessive sharpening when repeatedly optimized.

For our method the operative gradient is

\[
\nabla_\theta Q(R(y;\theta))
=J_R(\theta)^T\nabla_xQ(x),
\qquad x=R(y;\theta).
\]

The projection through \(J_R\) matters. A poor unrestricted pixel direction may be harmless when the allowed edits are tightly restricted; a seemingly smooth score can still misrank the few directions our filters permit. We must test **parameter-space trajectories**, not just pixel-gradient existence.

The tables give an evidence-based suitability assessment, not a local gradient audit. No listed NR model has been established here as a universally stable, meaningful objective for our operator family. “Differentiable” below assumes tensor preprocessing, enabled autograd, and no image serialization or hard decision between the model and input.

### 5.2 Full-reference candidates

| Model / source | Reference, training data and objective | Differentiability and optimization evidence | Generalization / failure concerns | Recommended role |
|---|---|---|---|---|
| **LPIPS — CVPR 2018** [paper and code](https://github.com/richzhang/PerceptualSimilarity) | Requires a reference. Deep features calibrated on BAPPS human pairwise/JND judgments; pretrained recognition backbone | Input gradients supported; repository discusses use as a loss. Backbone choice affects behavior | Perceptual invariance and adversarial directions permit pixel/structure disagreement; task-dependent loss performance | Independent evaluation with clean targets; weak anchor to a conservative reference if its limitations are explicit |
| **DISTS — TPAMI 2022** [paper](https://www.cns.nyu.edu/pub/lcv/ding20a-preprint.pdf), [code](https://github.com/dingkeyan93/DISTS) | Requires a reference. VGG-based structure/texture statistics; KADID-10k and texture data including DTD | Differentiable; published image-optimization evidence exists | Texture-resampling tolerance is intentional. Similar score can hide different fine details or small geometric changes | Evaluate perception separately from exact feature fidelity; do not use alone to protect texture identity |
| **PieAPP — CVPR 2018** [paper/code](https://github.com/prashnani/PerceptualImageError) | Requires a reference. Learned patch errors and weights from PieAPP pairwise preferences | CNN components differentiable; fix dense patch sampling for reproducible optimization | Prediction benchmark success does not guarantee stable optimization; patch coverage and unseen artifacts matter | Useful independent perceptual evaluator; secondary loss when a legitimate reference exists |
| **MS-SSIM / related structural metrics** [differentiable implementations and optimization study](https://github.com/dingkeyan93/IQA-optimization) | Reference required; handcrafted multiscale luminance, contrast and structural comparisons | Differentiable with numerical stabilization; established restoration-loss use | Can tolerate changes that matter for thin lines or texture; not a naturalness density | Low-cost structural comparator and controlled baseline |

**Using the noisy observation as the “clean reference” changes the problem.** LPIPS\((\hat x,y)\) or DISTS\((\hat x,y)\) measures similarity to a degraded image. It may discourage noise removal, tolerate wrong detail, or behave unpredictably outside the metric's intended comparison. It is an anchor heuristic, not a clean-image oracle.

### 5.3 No-reference IQA candidates

| Model / source | Main training signal and datasets | Differentiability / input-gradient status | Generalization and exploitation | Suitability for our optimization |
|---|---|---|---|---|
| **BRISQUE — TIP 2012** [paper](https://live.ece.utexas.edu/research/quality/brisque_journal.pdf) | Natural-scene statistics plus SVR, calibrated on human scores, originally LIVE | Standard pipeline is not a reliable end-to-end gradient loss; PyIQA marks no backward support | Narrow distortion/content assumptions; restoration can match statistics while losing detail | Keep as a cheap evaluation or derivative-free baseline |
| **NIQE — SPL 2013** [author implementation](https://github.com/utlive/niqe) | Fit statistics of pristine patches; no opinion labels | Common implementation has fitted statistics and nondifferentiable operations; PyIQA marks no backward | Domain, resolution, patch and content sensitivity; low score is not evidence of correctness | Historical baseline only; poor choice as the sole optimization objective |
| **MUSIQ — ICCV 2021** [paper](https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html) | Multiscale transformer; models trained on KonIQ-10k, SPAQ, PaQ2PiQ or AVA | Neural score can backpropagate with suitable preprocessing; no general restoration-gradient guarantee | Dataset/checkpoint domain matters; resizing and patch scale can hide local defects | Reasonable NR comparison and candidate selector after trajectory tests |
| **MANIQA — CVPR NTIRE workshop 2022** [paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/html/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.html), [code](https://github.com/IIGROUP/MANIQA) | Attention-based IQA; PIPAL restoration-quality challenge, plus benchmark-specific training | Differentiable neural graph; crop sampling can add gradient variance | Explicitly included in adversarial robustness studies; weights trained on KonIQ are not equivalent to PIPAL weights | Relevant restoration-aware candidate; use only with independent fidelity control |
| **TOPIQ — TIP 2024** [paper](https://arxiv.org/abs/2308.03060) | Cross-scale feature attention; NR variants for KonIQ/SPAQ/FLIVE; separate FR and aesthetic variants | Tensor implementation supports gradients; choose the NR checkpoint deliberately | Content/domain bias; attacked in Guardians. Strong correlations do not certify optimized outputs | Good first compact modern NR candidate, subject to tests |
| **CLIP-IQA / CLIP-IQA+ — AAAI 2023** [paper](https://arxiv.org/abs/2207.12396), [code](https://github.com/IceClear/CLIP-IQA) | CLIP with opposing quality prompts; “+” adds learned prompts using IQA labels such as KonIQ | Image encoder and soft similarity score are differentiable | Prompt sensitivity, semantic/aesthetic bias; CLIP-IQA+ appears in adversarial testing | Useful complementary score, weak choice as the only prior |
| **LIQE — CVPR 2023** [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Blind_Image_Quality_Assessment_via_Vision-Language_Correspondence_A_Multitask_Learning_CVPR_2023_paper.html), [code](https://github.com/zwx8981/LIQE) | Joint quality, scene and distortion learning from LIVE, CSIQ, KADID-10k, BID, CLIVE and KonIQ | Neural expected score differentiable; control stochastic crops | Broad training and gMAD testing support comparison ability; neither proves white-box robustness | Good diverse secondary candidate; assess local ranking within our filter family |
| **CONTRIQUE — TIP 2022** [paper](https://arxiv.org/abs/2110.13266), [code](https://github.com/pavancm/CONTRIQUE) | Contrastive feature learning from synthetic/authentic distortions; subsequent quality regressor uses MOS | Encoder plus tensor linear head is differentiable; NumPy/SVR-style wrappers may break gradients | Representation learning is unsupervised, but the final score is not generally label-free; unseen artifact bias remains | Useful independent representation/score comparator; audit the scoring wrapper |
| **FGResQ — AAAI 2026** [paper](https://arxiv.org/abs/2508.14475), [code](https://github.com/sxfly99/FGResQ) | Restoration-specific FGRestore: 18,408 restored images across six tasks and 30,886 preference pairs; scoring plus fine ranking | Neural single-image and pair-ranking heads can support gradients in a tensor implementation; trajectory stability unestablished | Training distribution is closer to restoration, but adversarial optimization and new filter families remain open | Especially relevant for candidate ranking; promising experiment rather than a validated safe loss |

**Implementation warning:** the checked [PyIQA model card](https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md) explicitly labels BRISQUE and NIQE as having no backward support. Its neural models have different variants, input ranges, crops, and score directions. The [repository](https://github.com/chaofengc/IQA-PyTorch) also distinguishes inference from loss use. Freeze model weights while retaining derivatives with respect to the image; freezing weights alone does not block those derivatives.

For BRISQUE or NIQE, a low-dimensional search can still use finite differences, grids, or derivative-free optimizers. This does not remove score exploitation or make discontinuous estimates reliable gradients.

### 5.4 Multimodal models, aesthetics, and learned priors

| Candidate / source | Training and reference requirements | Differentiability and limitations | Best role |
|---|---|---|---|
| **NIMA / aesthetic scorers** [NIMA](https://arxiv.org/abs/1709.05424) | AVA aesthetic model uses human rating distributions; no reference at inference | Expected rating is differentiable. Aesthetics can favor composition, contrast or saturation unrelated to restoration truth | Weak optional appearance preference; avoid using aesthetics as the primary denoising prior |
| **Q-Align — ICML 2024** [paper/project](https://q-align.github.io/) | Quality-level supervision; IQA training includes KonIQ, SPAQ, KADID; related aesthetic/video tasks use other datasets | Expected score from level-token probabilities can differentiate through local model logits. Generated text, argmax and remote APIs do not provide that path | Expensive independent judge or candidate ranker; direct iterative use requires a graph and cost audit |
| **DeQA-Score — CVPR 2025** [paper](https://openaccess.thecvf.com/content/CVPR2025/html/You_Teaching_Large_Language_Models_to_Regress_Accurate_Image_Quality_Scores_CVPR_2025_paper.html), [code](https://github.com/zhiyuanyou/DeQA-Score) | Soft quality distributions and ranking objectives; KonIQ/SPAQ/KADID, with broader out-of-distribution tests | Expected token score is differentiable in principle; actual wrapper must preserve tensor processing. Rating uncertainty is not pixel reconstruction uncertainty | Strong modern evaluation candidate; optimize only after adversarial and trajectory tests |
| **CLIP / DINO embeddings** [CLIP](https://proceedings.mlr.press/v139/radford21a.html), [DINO](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) | CLIP learns image-text correspondence; DINO learns visual representations without class labels | Differentiable features, but no inherent scalar naturalness score. Semantic invariance may ignore small evidence-critical changes | Auxiliary feature preservation at reliable scales; do not treat embedding norm or similarity as image likelihood |
| **Pretrained discriminator / critic** [adversarial regularizers](https://papers.nips.cc/paper_files/paper/2018/hash/d903e9608cfbf08910611e4346a0ba44-Abstract.html) | Depends on clean domain and negative/generator distribution | Differentiable, but saturation, extrapolation and negative-distribution bias can make gradients unhelpful | Domain-specific ablation; explicit regularizer training is preferable to borrowing an unrelated critic |
| **Explicit learned energy / score prior** [WCRR](https://doi.org/10.1137/23M1565243), [SNORE](https://proceedings.mlr.press/v235/renaud24a.html) | Clean-image or denoising training; no inference reference | Has a more explicit mathematical interpretation than an opinion score; still sensitive to domain and operator mismatch | Strong alternative to NR-IQA for guiding a restricted deterministic solver |

The “fidelity loss” used in some IQA training refers to fidelity of predicted preference probabilities to human labels. It should not be confused with \(\|H\hat x-y\|\), physical consistency, or scene identity.

### 5.5 My candidate selection

For a first controlled experiment, compare:

1. **TOPIQ-NR** as a practical modern score.
2. **FGResQ** for restoration-specific ranking.
3. **LIQE** as a different representation and training approach.
4. **A compact explicit energy such as WCRR** as the probabilistically/variationally motivated alternative.
5. **NIQE/BRISQUE** as historical controls, reflecting the current project direction.

Keep MUSIQ or DeQA-Score outside the optimization loop for independent evaluation. LPIPS and DISTS require clean references for their intended full-reference use. This selection is a proposed experimental design; it is not a claim that the selected models already have trustworthy optimization gradients.

## 6. Failure modes and negative evidence

### 6.1 Findings supported by experiments

| Evidence | What it tells us | Consequence for this project |
|---|---|---|
| IQA-as-loss comparison, core paper 25 | Evaluation accuracy does not reliably predict optimization performance; outcomes vary by restoration task | Test the optimized outputs, not only score correlations |
| Guardians, core paper 26 | Modern NR-IQA scores can be manipulated; defenses also have costs | Include deliberate score-maximizing attacks and independent evaluation |
| LPIE, core paper 5 | Excessive learned-quality weight produces distortions | Bound the role of the quality term and log quality–fidelity trajectories |
| ProxyOpt, core paper 7 | Optimizing a downstream objective can yield visually unusual processing | A successful optimizer can expose a misaligned objective |
| TTPO, core paper 27 | More aggressive diffusion editing can alter structure | Parameter limits are important, but visual similarity is weaker than measurement evidence |
| Hallucination analysis and HalluGen, core papers 22/24 | Realistic errors can persist despite measurement consistency or under distribution shift | Evaluate false content and operator/domain mismatch explicitly |

These observations do not establish that every model fails in every setting. They establish a concrete need to test each scorer in the intended optimization loop.

### 6.2 Mechanisms to test

**Oversharpening and ringing.** A score that rewards crisp edges can favor halos or oscillations. Sharpened noise can resemble detail. Inspect step-edge overshoot and flat-region false edges, not just global quality.

**Texture replacement.** Different textures may have similar patch distributions or perceptual features. A reference-tolerant metric can prefer the wrong individual texture realization. Evaluate locations, phase-sensitive features and thin structures.

**Semantic drift.** Foundation-model features can preserve the broad concept while changing a character, a small object, a count, or a facial feature. Text similarity cannot establish spatial identity.

**Distribution bias.** A natural-image prior may suppress an unusual but real structure. Medical images, microscopy, technical drawings, and text are especially useful stress tests because their relevant details differ from generic photo aesthetics.

**Metric search overfitting.** Even selecting from a finite set can exploit score errors. Increasing the number of candidates may raise the winning score without improving truth or human preference. Measure performance as the search budget grows.

**Surrogate exploitation.** If \(\widetilde R\) approximates a classical operator, the optimizer can find settings where its approximation is inaccurate. Every accepted setting should be evaluated through the actual operator.

**Shared failure across metrics.** Several IQA models may share a backbone, training images, or annotation bias. Agreement is useful evidence of ranking stability, but is not independent confirmation of physical fidelity.

**Blind-model collusion.** Jointly optimizing an image and an unconstrained degradation can reduce residuals by changing the explanation rather than recovering the image. Restrict blur kernels, noise models and camera parameters physically, and evaluate operator estimation separately.

### 6.3 A useful optimization audit

Before a metric becomes a loss, run small parameter sweeps and short optimization trajectories on images with known clean references. For every step record the optimized score, unseen IQA scores, PSNR, LPIPS/DISTS, measurement residual, false/missing edges, and operator parameters. Inspect the images at native resolution.

Also test finite differences against autodiff, perturb initial parameters, change crop locations and seeds, vary resolution, and optimize far beyond the intended stopping point. The last test reveals what the objective rewards when its weaknesses become accessible. Keep those attack images out of the final held-out evaluation if they influence method design.

## 7. Closest precedents and novelty assessment

### 7.1 What is already established?

| Component of our idea | Closest precedents | Remaining distinction to demonstrate |
|---|---|---|
| Tune a denoiser without a clean test target | SURE-LET; Zhu–Milanfar; Noise2Self | Whether learned quality adds value beyond estimated reconstruction risk |
| Combine a learned prior with data fidelity | EPLL; adversarial regularizers; PnP/RED; WCRR | Whether a restricted operator space yields stronger practical or mathematical control |
| Use a frozen quality model in enhancement | LPIE; modern preference-guided restoration | Per-image parameter tuning, explicit physical constraints, and independent fidelity testing |
| Optimize classical processing parameters through a neural approximation | ProxyOpt; instance-specific surrogate learning | Reference-free inference, no learned parameter predictor, and protection against proxy/metric exploitation |
| Learned guidance with interpretable operators | Exposure; TNRD | Evidence-based restoration rather than primarily appearance or supervised reconstruction |
| Use a frozen generative prior in test-time optimization | Graikos; RED-diff; SNORE-related score regularization | Restrict updates to an interpretable operator family and quantify unsupported change |
| Quantify unsupported content and uncertainty | Bhadra; image-to-image uncertainty; HalluGen | A natural-image/filter-tuning protocol with calibrated, task-relevant risk |

### 7.2 Verdict

**The broad strategy is well established; our exact combination is partially explored.** I did not verify an existing paper implementing every proposed component in the same form. That is not enough to claim novelty: recombining a quality score, a filter bank and a data term is a small change unless it produces a substantive mechanism, analysis, or experimental finding.

Potentially defensible differences are:

- A statistically controlled candidate set in which a learned score can only break near-ties.
- Bounds on what the optimizer can alter in observed, weak and unobserved components.
- A training/evaluation protocol for quality models specifically as restoration optimizers.
- Calibrated uncertainty or abstention for structures, rather than an unqualified sharp image.
- A clear fidelity–perception advantage at comparable compute, with robustness to both model mismatch and intentional metric exploitation.

**A weak novelty claim:** “we optimize classical filters using an AI quality model.”

**A stronger possible claim:** “we introduce a constrained selection procedure that improves perceptual preference within a calibrated reconstruction-risk budget, and reduces false structures under specified noise and operator shifts.” This would still require proof or convincing experiments; the review does not establish that it already works.

For the current spectral/ensemble design, a globally learned weighting on validation crops is a different method from per-image test-time tuning. An image–filter ranking matrix or HITS-style weighting alone does not give either reconstruction-risk control or a likelihood interpretation. Those distinctions should be explicit in the eventual paper.

## 8. Stronger mathematical formulations

The following are proposed formulations informed by the review. They are not claims of experimentally demonstrated performance or proven novelty.

### 8.1 Basic constrained deterministic restoration

Let \(x_0=R(y;\theta_0)\) be a conservative baseline. Define a restricted feasible parameter set

\[
\Theta_{safe}(y)=\left\{\theta\in\Theta:
L_{meas}(R(y;\theta),y)\leq\tau,
\ \|W(R(y;\theta)-x_0)\|_2\leq r,
\ \|\theta-\theta_0\|_D\leq s\right\}.
\]

Then choose

\[
\theta^*\in\arg\max_{\theta\in\Theta_{safe}(y)}Q(R(y;\theta)).
\]

- \(\Theta\): parameter ranges justified by the operators and imaging model.
- \(\tau\): measurement tolerance based on the likelihood or calibration.
- \(W\): weights on changes to reliable or sensitive features.
- \(r,s\): image-space and parameter-space trust regions.

This prevents the quality score from trading unlimited fidelity for score improvement. It still only guarantees the constraints actually imposed. If the feasible set is empty, report the inconsistency or use a separately specified fallback; silently relaxing constraints defeats the purpose.

For a noiseless consistent linear problem, an explicit measurement projection is

\[
\Pi(z)=z+H^\dagger(y-Hz).
\]

For noisy measurements, one can instead solve a proximal data step:

\[
x^+=\arg\min_x\tfrac12\|Hx-y\|_{\Sigma^{-1}}^2
+\tfrac1{2\eta}\|x-z\|_2^2
=\left(H^T\Sigma^{-1}H+\eta^{-1}I\right)^{-1}
\left(H^T\Sigma^{-1}y+\eta^{-1}z\right).
\]

These are different operations with different noise assumptions. A pixel-space correction can also leave the original family \(\{R(y;\theta)\}\). Either enforce constraints while optimizing \(\theta\), or explicitly include the correction as a documented stage of \(R\).

### 8.2 Better for AWGN: risk-constrained quality selection

For \(y=x+n\), \(n\sim\mathcal N(0,\sigma^2I)\), a fixed weakly differentiable estimator has SURE

\[
\widehat{\mathcal R}_{SURE}(\theta;y)=
\frac{\|R(y;\theta)-y\|_2^2}{N}
+\frac{2\sigma^2}{N}\operatorname{div}_y R(y;\theta)
-\sigma^2.
\]

Unlike residual MSE alone, the divergence term accounts for estimator sensitivity to noise. The original statistical basis is [Stein, Annals of Statistics 1981](https://www.stat.yale.edu/~hz68/619/Stein-1981.pdf); SURE-LET provides the denoiser-combination precedent.

Start with

\[
\theta_0\in\arg\min_{\theta\in\Theta}\widehat{\mathcal R}_{SURE}(\theta;y)
\]

and permit quality selection only among near-optimal candidates:

\[
\max_{\theta\in\Theta}Q(R(y;\theta))
\quad\text{subject to}\quad
\widehat{\mathcal R}_{SURE}(\theta;y)
\leq\widehat{\mathcal R}_{SURE}(\theta_0;y)+\epsilon,
\]

with additional trust-region and structure restrictions if justified.

**The unresolved statistical issue is selection.** Unbiasedness for each fixed estimator does not make the minimum of many estimated risks unbiased. When \(\theta\) is itself selected from \(y\), the full adaptive estimator has additional dependence on \(y\). Do not label the displayed constraint a true-risk guarantee without a selection-aware argument or calibration of the complete selection rule.

A Monte Carlo divergence estimate can use

\[
\operatorname{div}R(y)\approx
\frac{1}{K\delta}\sum_{k=1}^K
z_k^T\big[R(y+\delta z_k)-R(y)\big],
\quad \mathbb E[z_kz_k^T]=I.
\]

Fix probes across candidate comparisons to reduce comparison noise; study sensitivity to \(K\) and \(\delta\). Discontinuous operators require particular care. Validation must cover the entire adaptive pipeline, including candidate search and stopping. For general \(H\), generalized risk estimates may concern only observable components; they do not reveal null-space error for free.

### 8.3 Self-supervised alternative

For a held-out component set \(J\), let \(R_J(y_{J^c};\theta)\) predict pixels without observing their noisy values. Under suitable independent, unbiased noise,

\[
\mathbb E\|R_J(y_{J^c};\theta)-y_J\|^2
=\mathbb E\|R_J(y_{J^c};\theta)-x_J\|^2
+\mathbb E\|n_J\|^2.
\]

This gives a reconstruction-oriented selection signal. **The parameter selector must not leak held-out noisy values back into their own prediction.** Merely making each fixed-parameter filter blind is insufficient if its parameters are chosen using the same held-out values. Use a nested or properly cross-fitted procedure and evaluate the complete rule. Correlated camera noise also changes the required masking. [Noise2Self](https://proceedings.mlr.press/v97/batson19a.html).

### 8.4 Restricted convex filter ensembles

For fixed candidate outputs \(d_j(y)\),

\[
R(y;w)=\sum_{j=1}^J w_jd_j(y),
\qquad w_j\geq0,\quad\sum_jw_j=1.
\]

The output lies in the convex hull of the candidates and, at each pixel/channel, within their minimum and maximum. This is an inspectable restriction. It does not prevent all false patterns, and candidates may all erase the same detail.

A useful first design is a small global weight vector with a trust region around a risk-selected baseline. Per-pixel weights create much more expressive images and substantially increase the possibility of metric exploitation. If spatial weights are later added, limit their resolution, variation, and number of degrees of freedom.

This also makes a critical diagnostic easy: compare the best attainable clean-reference result within the candidate family with the selected result. If even the oracle cannot perform well, changing the scorer cannot solve the family's expressivity problem. The oracle belongs only in analysis, never in deployed selection.

### 8.5 Frequency- and operator-aware control

Let \(H=USV^T\). A change \(\Delta x\) in direction \(v_i\) changes the measurement by \(s_i\langle v_i,\Delta x\rangle\). Large singular values make changes observable; small values make inference noise-sensitive; zero values are invisible.

One possible parameterization is

\[
R(y;\theta)=x_0+\sum_b a_b(\theta)\,B_b r(y),
\]

where \(B_b\) are fixed bands and \(r(y)\) is an evidence-derived residual or correction. Bound each gain using noise and operator sensitivity. Restrict uncertain directions more strongly if our aim is conservative reconstruction, even though a purely perceptual method might fill them aggressively.

For our complementary spectral decomposition, preserve the exact reconstruction identity of the bands and control deviations from the baseline gains. Merely preserving average brightness or total spectral magnitude leaves many possible structural errors. Fourier phase and spatial localization matter.

For plain denoising, \(H=I\) has equal singular values; the singular spectrum alone provides no band reliability ranking. Use the noise spectrum and validated signal-reliability estimates. For blur and downsampling, operator-specific weak modes become essential.

### 8.6 An expanded objective with explicit meanings

\[
\begin{aligned}
L(\theta)= {}&\lambda_m L_{meas}(R(y;\theta),y)
-\lambda_q Q(R(y;\theta))\\
&+\lambda_s\|W_s[\Phi(R(y;\theta))-\Phi(x_0)]\|_1\\
&+\lambda_f\|W_f\mathcal F(R(y;\theta)-x_0)\|_2^2\\
&+\lambda_c\|R(y;\theta)-x_0\|_2^2
+\lambda_\theta\|\theta-\theta_0\|_D^2.
\end{aligned}
\]

| Term | Practical interpretation | Important limitation |
|---|---|---|
| Measurement | Correct likelihood for the observed image | Cannot determine null-space content |
| Naturalness | A tested IQA score, explicit energy, or denoiser-derived prior | Can be exploited or biased |
| Structure | Reliable gradients, landmarks, or selected feature maps | A noisy or biased anchor can lock in errors |
| Frequency | Penalize changes in specified modes, including phase-sensitive complex differences | Band selection must reflect the operator and noise |
| Conservative change | Keep output near a risk-oriented baseline | Can limit useful restoration |
| Parameter trust region | Restrict the magnitude of changes to interpretable controls | Parameter distance may not directly bound image change |

The features \(\Phi\) should serve a stated task. DINO features may help preserve broad layout, but cannot replace pixel-level checks for small structures. Identity losses are relevant for faces, not a general denoising certificate. Spatial confidence weights must be validated rather than inferred solely from a quality model's self-confidence.

Prefer hard constraints for essential requirements; use the weighted objective to explore secondary trade-offs. There is no universally correct collection of \(\lambda\)'s. A Pareto curve is more informative than a single favorable setting.

### 8.7 Robust quality scoring and stopping

Possible defenses to test include:

- Score several fixed patches/scales and use a conservative aggregate; retain full-image checks to catch seams or global artifacts.
- Normalize different metrics using held-out calibration images; preserve their score directions.
- Penalize disagreement between genuinely different scorers, while recognizing shared biases.
- Average over small transformations that preserve the intended content, then test adaptive attacks against the same averaging.
- Require minimum improvement beyond scorer variance before accepting an update.
- Limit candidate count, iterations, and gain changes; keep the baseline when evidence for improvement is weak.

An illustrative score is

\[
Q_{rob}(x)=\frac1K\sum_k\widetilde Q_k(x)
-\beta\,\mathrm{Std}_k[\widetilde Q_k(x)].
\]

This is an engineering heuristic, not a calibrated probability or proof against reward hacking. Its value must be assessed against single-model optimization at the same search budget.

### 8.8 An even more conservative endpoint: recover features

For a task-relevant quantity \(f(x)\), consider its range over a defensible image set:

\[
\left[
\inf_{x\in\mathcal C(y)}f(x),
\sup_{x\in\mathcal C(y)}f(x)
\right].
\]

If the interval spans both “feature absent” and “feature present,” the observation does not support a confident claim. Tightening the set with a learned prior adds assumptions that must be stated. Global extrema may also be computationally difficult; a few local optimizations do not certify the full range.

For publishable early work, use simple measurable features—edge contrast, thin-line presence, intensity difference—where useful bounds are possible. This is closer to category 3 than maximizing photo naturalness everywhere.

## 9. Experimental plan for this project

### 9.1 Use the existing images without forcing a common full-image size

DIV2K train/validation, CBSD68 and Kodak have different image dimensions. That is normal for restoration. Use fixed-size crops for fitting or calibration and native-resolution images for final testing, with documented padding or tiling. Resizing every image would change the image statistics and the degradation being studied. The recorded PNG sizes in MB describe storage; they do not identify a noise model.

Use DIV2K training images for development and any learned prior calibration, DIV2K validation for freezing design choices, and CBSD68/Kodak for held-out reporting. If earlier experiments have already influenced decisions using a test set, disclose that and add an untouched holdout. Split by original image, not by patch, so crops from the same image cannot leak across partitions.

The current experiment notes describe global weights learned from validation crops. Treat those weights as a fitted model; do not call that configuration “no training data.” A pretrained IQA model also carries external supervision even if the restoration parameters are fitted only at test time.

### 9.2 Start with one identifiable question

**Primary question:** at AWGN \(\sigma=25/255\), does learned quality guidance improve perceived results without materially increasing reconstruction error or false structures relative to risk-selected deterministic filtering?

Use RGB float images with a documented range and channel convention. Generate noise with reproducible seeds. Keep the unclipped AWGN experiment distinct from clipped/quantized noise, because clipping changes the likelihood and affects SURE. Add \(15/255\) and \(50/255\) after the primary experiment is stable.

For each test image, use at least five fixed noise realizations when practical. Compare methods on identical observations. Average per-image PSNR according to the chosen benchmark convention, and report both across-image and noise-seed variability. Bootstrap images or use a hierarchical procedure; treating every pixel as an independent sample exaggerates confidence.

### 9.3 Minimum comparison set

| Group | Required comparisons | Purpose |
|---|---|---|
| Current family | Individual filters; fixed average; current global learned weights | Establish whether combining filters helps |
| Risk-only tuning | SURE-selected parameters or a correctly masked self-supervised selector | Isolate learned quality's added value |
| Quality-only tuning | NIQE/BRISQUE and modern NR-IQA with equal search budgets | Reveal the uncontrolled objective's behavior |
| Proposed controls | Risk constraints, trust regions, structure/frequency controls and fallback | Attribute improvements to actual mechanisms |
| Classical strong baseline | BM3D / color BM3D | Avoid comparisons limited to weak filters |
| Learned distortion baseline | DRUNet or Restormer with the correct noise/task checkpoint | Establish the quality attainable with strong supervised priors |
| Generative comparison | One measurement-aware method appropriate to the task | Quantify fidelity/perception/compute trade-offs |
| Analysis-only oracle | Best candidate or weights chosen using clean targets | Separate scorer failure from operator-family limitations |

Baseline resources: [BM3D](https://webpages.tuni.fi/foi/GCF-BM3D/index.html), [DPIR/DRUNet](https://github.com/cszn/DPIR), [Restormer](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html). These are established reference baselines, not a claim about the single best 2026 model on every task.

If a pretrained baseline saw benchmark images during training, that matters. Record checkpoint provenance, training data, preprocessing, RGB versus luminance evaluation, padding, precision, and clipping. Include the quality model's cost when claiming that the deterministic method is lightweight.

### 9.4 Evaluate at least four different outcomes

1. **Reconstruction:** PSNR/MSE and SSIM on known clean targets; per-image distributions and worst-tail failures.
2. **Perception:** LPIPS/DISTS against clean references, independent NR-IQA, and blinded human comparisons where feasible. Do not report only the optimized metric.
3. **Evidence preservation:** false and missing edges, line/character changes, spatial error maps, measurement residuals, and sensitivity to controlled input perturbations.
4. **Cost and reliability:** latency, number of scorer evaluations, peak memory, optimization failures, and fraction of images that trigger fallback.

Keep perceptual preference questions separate from fidelity questions in a human study. People can prefer sharper but incorrect details. For fidelity judgments, provide appropriate reference or measurement evidence and a precisely stated comparison task.

Small datasets such as Kodak are useful paired benchmarks but poor support for sweeping distributional conclusions. In particular, a FID estimate on 24 images is not a convincing primary endpoint.

### 9.5 Stress tests that address hallucinations directly

Use synthetic fixtures with known truth: blank regions, isolated edges, thin lines, repeated gratings, small text, and near-resolution-limit structures. Place them in both simple and natural backgrounds. Measure additions and deletions separately; a method that removes every fine detail may have few false additions but fail restoration.

Test:

- Unseen noise levels and spatially correlated noise.
- Incorrect supplied noise variance.
- Blur-kernel mismatch and downsampling mismatch.
- Saturation, clipping, JPEG and mixed degradations as separately labeled settings.
- Small structured perturbations designed to raise the quality score.
- Increasing optimization budget and starting from different parameter settings.
- New image domains and rare content.

After denoising is established, add one known-kernel deblurring task and one defined SR task. These expose weak/null-space ambiguity that denoising alone does not. Avoid claiming “general restoration without hallucinations” from AWGN results alone.

### 9.6 The visualizations to produce

Use ordinary **Matplotlib or Plotly scripts**, saving reproducible figures:

- Pareto scatter: reconstruction error versus independent perceptual score; color by false-structure rate.
- Optimization trajectory: selected IQA score, PSNR, residual and parameter values versus iteration.
- Oracle gap: best achievable candidate quality versus the chosen candidate.
- Robustness curves: performance versus noise/operator mismatch and search budget.
- Native-resolution crops: clean, observed, conservative baseline, proposed result, absolute error, and false/missing-edge overlays.

These figures test the research claim more directly than additional histograms of file sizes.

## 10. Five research directions and reading order

### Direction 1 — Quality-guided denoising within a reconstruction-risk budget

**Proposal:** a small interpretable filter ensemble; SURE or correctly constructed self-supervision defines an admissible candidate set; NR-IQA selects within it; uncertain cases return a baseline.

**Potential contribution:** selection-aware risk control or calibrated acceptance combined with a measurable perceptual benefit.

**Closest work:** SURE-LET, Zhu–Milanfar, Noise2Self, ProxyOpt.

**Critical experiment:** beat risk-only selection in blinded preference at matched reconstruction risk, with fewer failure cases than quality-only selection. If adding IQA has no consistent benefit, the negative result should change the method rather than be hidden by metric choice.

**Feasibility:** highest for the current project.

### Direction 2 — Restrict learned prior guidance by operator observability

**Proposal:** bounded bandwise corrections with explicit treatment of reliable, weak and null measurement components. Use a frozen learned energy or scorer only within those bounds.

**Potential contribution:** a useful image-change or sensitivity bound, and reduced false detail under known and mildly mismatched operators.

**Closest work:** variational regularization, DDNM, Prox-PnP, WCRR.

**Critical experiment:** deblurring/SR with thin structures and controlled kernel errors; report retained true detail alongside false additions. Merely freezing all high frequencies would not establish a good solution.

**Feasibility:** moderate; requires careful operator mathematics.

### Direction 3 — Train or calibrate a quality model for optimization trajectories

**Proposal:** generate candidate outputs and deliberately exploited examples from the actual deterministic pipeline. Learn or calibrate preferences on these trajectories, with fidelity-aware rejection labels.

**Potential contribution:** demonstrate that optimization-specific training gives better parameter directions and fewer artifacts than a model with better ordinary IQA correlations.

**Closest work:** IQA optimization comparison, Guardians, FGResQ, learned parameter surrogates.

**Critical experiment:** hold out both images and operator families; test adaptive attacks and independent human judgments. Improvements only on the operators used for training would be insufficient evidence of generalization.

**Feasibility:** moderate; annotation effort may become the main cost.

### Direction 4 — Restore with calibrated confidence in small structures

**Proposal:** accompany deterministic restoration with confidence for measurable features such as edge contrast, line presence, or character strokes; abstain from enhancing features with weak support.

**Potential contribution:** a precise calibration target tied to unsupported visual content, with useful restoration at a controlled error rate.

**Closest work:** image-to-image uncertainty quantification and hallucination-map research.

**Critical experiment:** evaluate coverage, abstention, correct-detail recovery and false-detail rate under matched and shifted data. A pixelwise variance map alone is not the contribution.

**Feasibility:** moderate to demanding; strongest fit to the user's category 3.

### Direction 5 — A compact explicit prior for classical parameter tuning

**Proposal:** compare a shallow learned patch/filter energy with generic IQA and pretrained discriminators while keeping the same deterministic operator family and fidelity constraints.

**Potential contribution:** an energy whose parameter-space gradients are bounded or otherwise analyzable, plus evidence that it generalizes better than quality ratings under optimization.

**Closest work:** EPLL, adversarial regularizers, WCRR, TNRD and score regularization.

**Critical experiment:** equalize operator family and optimization budget; separate benefits of the prior, constraints and solver. A smaller network alone does not establish novelty.

**Feasibility:** moderate; a promising alternative if NR-IQA proves easy to exploit.

### Open gaps worth investigating

- Reliable, selection-aware risk bounds for per-image perceptual parameter tuning.
- Metrics evaluated as optimizers on restricted processing families rather than as passive judges.
- Practical confidence in small natural-image structures beyond global PSNR.
- Separation of operator uncertainty, image-prior uncertainty and IQA disagreement.
- Evaluation protocols that report false additions and lost real detail together.
- Scorers that remain useful across native resolutions, crops and noise distributions.
- Fair computational comparisons including repeated quality-model evaluation.

### The ten-paper shortlist for our exact idea

Read these first, in roughly this order:

| Order | Paper | Why it belongs in the first reading pass |
|---|---|---|
| 1 | [Zhu–Milanfar, 2010](https://research.google/pubs/automatic-parameter-selection-for-denoising-algorithms-using-a-no-reference-measure-of-image-content/) | Closest conceptual predecessor to score-based denoiser tuning |
| 2 | [SURE-LET, 2007](https://bigwww.epfl.ch/preprints/blu0701p.pdf) | Statistical foundation and baseline for combining estimators |
| 3 | [ProxyOpt, 2019](https://www.cs.princeton.edu/~fheide/proxyopt) | Differentiable tuning of classical processing parameters |
| 4 | [Instance-specific surrogates, 2025](https://arxiv.org/abs/2407.17530) | Recent direct prior art for image-dependent BM3D control |
| 5 | [Learned Perceptual Image Enhancement, 2018](https://arxiv.org/abs/1712.02864) | Frozen quality objective and fidelity anchoring |
| 6 | [Adversarial Regularizers, 2018](https://papers.nips.cc/paper_files/paper/2018/hash/d903e9608cfbf08910611e4346a0ba44-Abstract.html) | Learned scalar prior inside inverse optimization |
| 7 | [WCRR, 2024](https://doi.org/10.1137/23M1565243) | Compact interpretable learned energy and convergence |
| 8 | [IQA optimization comparison, 2021](https://arxiv.org/abs/2005.01338) | How to evaluate a metric as a loss |
| 9 | [Guardians, 2025](https://proceedings.mlr.press/v267/gushchin25a.html) | Concrete evidence of metric exploitation |
| 10 | [Noise2Self, 2019](https://proceedings.mlr.press/v97/batson19a.html) | A reference-free fidelity-oriented selection alternative |

For the hallucination question itself, then read Bhadra, HalluGen, the two perception trade-off papers, and image-to-image uncertainty. For diffusion implementation choices, start with DDNM versus DPS, then DAPS/SITCOM and RED-diff.

### Recommended first research milestone

Build a small, reproducible study comparing **risk-only, quality-only, and risk-constrained quality selection over the same deterministic candidates**. Include an oracle analysis, multiple noise seeds, independent perception scores and false-structure tests.

The result should answer one concrete question: **does the quality prior contribute useful selection information after fidelity is protected, or does it mainly exploit the scorer?** That answer should determine the next method, before investing in a larger system.
