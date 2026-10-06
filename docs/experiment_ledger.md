# Experiment Ledger

This ledger records every meaningful experiment represented by the supplied scripts. “Observed result” means evidence that was actually recoverable. A printed claim embedded in code is not treated as an observed result.

## E01 Paper inspired deterministic self fusion

**Hypothesis.** A single image contains separable smooth, textural, and edge information that can be extracted by bilateral, Gabor, and derivative operations and recombined into a perceptually better image.

**Method.** `self_fuse_net.py`, `run_demonstration.py`, and the thesis implement bilateral low-frequency extraction; two-stage Sobel edge enhancement gated by VIF; Gabor candidate filtering gated by BRISQUE and NIQE; and a 1,331-point fusion-weight search.

**Implementation validity.** Partial. It omits the paper’s customized U-Net and upsampling stage, so it is not a full Self-FuseNet reproduction. It also changes important details: Sobel is computed from bilateral-filtered patches in the main path; the Gabor candidates are clipped to unsigned 8-bit images; accepted maps are averaged rather than summed; and fusion accepts then averages all candidates rather than selecting a documented optimum.

**Observed result.** No saved output or quantitative result was supplied. The implementation exists, but its performance is unresolved.

**Limitations.** BRISQUE/NIQE are applied to filter response maps that are not natural photographs; grid weights are not brightness-normalized; averaging all accepted candidates has no optimality guarantee; and the output remains at input resolution.

**Conclusion.** A useful historical baseline, not a verified denoiser and not a reproduction of the complete paper.

**Worth revisiting.** Yes, only as a controlled baseline with stable floating-point operators, explicit component scaling, and proper evaluation.

## E02 Original histogram threshold diagnosis

**Hypothesis.** The paper’s threshold based on the minimum histogram count isolates significant edges.

**Method.** `scratch_edge_test.py` implements the paper-style threshold and compares it with Otsu and mean-plus-standard-deviation masks on a noisy Lenna patch.

**Implementation validity.** The diagnostic is meaningful. In a 256-bin histogram, at least one empty bin is common, making `min(hist)` equal to zero. The resulting expression is dimensionally questionable because a histogram count is subtracted from a gradient magnitude.

**Observed result.** No generated figure was supplied. The code comments explicitly identify noise amplification and explore alternatives.

**Conclusion.** The paper threshold is conceptually defective in this implementation. This is a bad formulation, not merely a tuning failure.

**Worth revisiting.** No. Replace it with a dimensionally coherent robust threshold or learned mask, then benchmark it.

## E03 Bilateral preconditioning before Sobel

**Hypothesis.** Computing the gradient on a bilateral-smoothed image suppresses noise edges while preserving structural edges.

**Method.** `scratch_edge_test2.py` compares raw-image Sobel against bilateral-preconditioned Sobel with a mean-plus-1.5-standard-deviation threshold. `self_fuse_net.py` later adopts bilateral preconditioning, with stricter thresholds.

**Implementation validity.** Conceptually sound for additive noise, though the single patch, single noise draw, and channel-wise mask are insufficient to choose the threshold.

**Observed result.** No output figure was supplied. Adoption in later code proves the idea influenced the pipeline, not that it improved metrics.

**Conclusion.** Potentially good idea with inconclusive evaluation.

**Worth revisiting.** Yes, as an ablation within a seeded benchmark.

## E04 Exhaustive fusion candidate logging

**Hypothesis.** Candidates that simultaneously improve BRISQUE and NIQE over the noisy input define a useful feasible fusion set.

**Method.** `run_denoise_experiment.py` saves every fusion candidate, logs every candidate passing both gates, reports the candidate minimizing BRISQUE plus NIQE, and averages all accepted raw candidates.

**Implementation validity.** Weak. The clean reference is available but ignored for PSNR/SSIM; the sum of BRISQUE and NIQE has no calibrated meaning; the mean of accepted candidates need not pass either gate; and `algo1_enhance_high_freq` is called without the already computed bilateral image, causing it to use a Gaussian fallback.

**Observed result.** No CSV, JSON summary, or output image was supplied.

**Conclusion.** Inconclusive experiment with one implementation bug and invalid model-selection logic.

**Worth revisiting.** Only after replacing gate-and-average with predeclared scalar selection and full-reference evaluation on synthetic data.

## E05 CycleGAN clean-domain prior training

**Hypothesis.** A discriminator trained to distinguish clean from generated images can serve as a differentiable naturalness prior for deterministic restoration parameters.

**Method.** `datasets.py`, `cyclegan_models.py`, and `train_cyclegan.py` define unpaired RealBlur/SIDD plus synthetic data, ResNet or DAB CycleGAN generators, and two multiscale spectrally normalized PatchGAN discriminators.

**Implementation validity.** Partial. The adversarial, cycle, identity, and image-pool mechanics are recognizable. However, feature matching compares spatial feature maps of unrelated unpaired images, which is not a valid pointwise target. Early stopping monitors moving training generator loss rather than held-out validation. Dataset paths, counts, splits, seeds, and logs are absent. The motion-kernel construction degenerates near vertical angles.

**Observed result.** Epoch-50 and epoch-100 clean-discriminator checkpoints are recoverable. Thesis figures show visually successful and visibly failed/hallucinated CycleGAN examples. No training curve or held-out score is recoverable. The thesis claim of 60,000 noisy and 40,000 clean images cannot be verified from the dataset code.

**Conclusion.** A real trained artifact exists, but its calibration and generalization are unverified. The prior may be useful; its training experiment is not reproducible.

**Worth revisiting.** Yes, after building a dataset manifest, held-out validation, domain-specific clean set, and a calibration test.

## E06 Supervised three-scalar sanity check

**Hypothesis.** If three fusion coefficients cannot improve a synthetically blurred image even with clean-target losses, the representation is too weak; if they can, unsupervised loss design is the remaining problem.

**Method.** `supervised_sanity_check.py` optimizes low/mid/high scalar weights on one clean Lenna crop using 15 combinations of SSIM, VIF, MAE, MSE, adversarial, and discriminator-feature losses.

**Implementation validity.** Useful in principle but narrow in execution. It uses one image, blur without noise, 1,000 steps, no repeated seeds, and no coefficient constraints. Final logged metrics are computed immediately before the last optimizer update.

**Observed result.** No printed table or saved images were supplied.

**Conclusion.** Scientifically meaningful diagnostic design, inconclusive execution evidence.

**Worth revisiting.** Yes, as a representational-ceiling test over a dataset and multiple degradation levels.

## E07 Discriminator only three-scalar optimization

**Hypothesis.** A frozen clean discriminator can directly optimize fusion weights without BRISQUE/NIQE grid search.

**Method.** `unsupervised_optimization.py` decomposes the image into Gaussian low, selected-Gabor mid, and residual high bands, then optimizes three scalar parameters using only least-squares adversarial loss.

**Implementation validity.** Invalid as written. The bands sum to the input, but coefficients are forced to sum to one; equal initialization therefore returns approximately one third of the input. The optimizer begins from a severely darkened image. No fidelity or mean constraint is used in this script, and discriminator-only optimization is vulnerable to prior exploitation.

**Observed result.** No output or metric log was supplied.

**Conclusion.** Potentially useful idea with a decisive implementation defect. Results from this exact script are unreliable.

**Worth revisiting.** Yes. The clean project corrects the identity parameterization and keeps the method as an explicitly unverified baseline.

## E08 MSE SSIM adversarial random degradation benchmark

**Hypothesis.** Discriminator-guided optimization can match or beat clean-target scalar optimization for some degradations.

**Method.** `benchmark_experiment.py` runs 100 random degradations of one Lenna crop and compares MSE, SSIM, and adversarial objectives.

**Implementation validity.** Invalid as a benchmark. All three runs are checkpoint-selected by lowest BRISQUE, the degradation RNG is not seeded, the same image is reused, the flawed normalized-coefficient model is imported, and “adversarial win” is defined by an OR across SSIM and BRISQUE.

**Observed result.** No CSV or saved wins were supplied.

**Conclusion.** Inconclusive and selection-biased.

**Worth revisiting.** No in its present form. The question is useful, but the benchmark must be rebuilt.

## E09 Fixed BM3D comparison

**Hypothesis.** The discriminator-guided scalar model can outperform BM3D.

**Method.** `benchmark_bm3d.py` extends E08 with BM3D using `sigma_psd=0.1` on every degradation.

**Implementation validity.** Invalid comparison. BM3D is given the same fixed Gaussian-noise parameter for Gaussian blur, motion blur, Poisson noise, and salt-and-pepper noise. It is not a deblurring method. The proposed method still carries the coefficient defect and BRISQUE selection bias.

**Observed result.** No CSV or saved wins were supplied.

**Conclusion.** Any claimed win against BM3D from this script is unreliable.

**Worth revisiting.** Yes, with noise-matched BM3D only on its valid domain and separate baselines for other degradations.

## E10 Satellite image comparison

**Hypothesis.** On a real satellite image without ground truth, the discriminator-guided output has better BRISQUE/NIQE than the input and BM3D.

**Method.** `compare_satelite.py` compares one `satelite_sample.png`, fixed BM3D, and adversarial-only scalar fusion.

**Implementation validity.** Weak. The image is absent, the noise level is unknown, BM3D is untuned, the coefficient defect remains, and two NR metrics are the only evaluation.

**Observed result.** No images or console table were supplied.

**Conclusion.** Inconclusive.

**Worth revisiting.** Only as a qualitative real-data case after the synthetic benchmark is sound.

## E11 Whole image versus four nonoverlapping patches

**Hypothesis.** Local fusion weights handle spatially varying image statistics better than one global set.

**Method.** `comprehensive_noise_experiment.py` compares whole-image optimization with a 2x2 split on satellite and Lenna images under four synthetic degradations.

**Implementation validity.** The hypothesis is good. The implementation creates hard patch boundaries, uses only BRISQUE for reporting, uses no seeds, and gives fixed-parameter BM3D every degradation.

**Observed result.** No result table or images were supplied.

**Conclusion.** Potentially good idea with inadequate stitching and evaluation.

**Worth revisiting.** Yes. This motivated the later overlap-add version.

## E12 Loss ablation

**Hypothesis.** Input fidelity and smoothness regularization stabilize the adversarial prior.

**Method.** `loss_ablation_study.py` compares adversarial-only, adversarial plus L1, SSIM, or TV, and nonadversarial L1+TV on one noisy satellite image.

**Implementation validity.** Invalid for quantitative ablation. TV is a sum rather than a mean and therefore has a resolution-dependent scale that can dominate the other losses. L1 and SSIM target the noisy input rather than a degradation-consistency model. The clean original exists but PSNR/SSIM to it are not reported.

**Observed result.** No table or outputs were supplied.

**Conclusion.** The regularization question is useful; the tested weights cannot support a conclusion.

**Worth revisiting.** Yes, with normalized losses and full-reference synthetic evaluation.

## E13 Patch L1 and mean constraint

**Hypothesis.** Adding input L1 fidelity and a mean-brightness constraint prevents discriminator-only collapse while preserving locally adaptive denoising.

**Method.** `final_unsupervised_benchmark.py` compares 2x2 patch optimization with adversarial+L1 against the same plus mean constraint on two images and four degradations.

**Implementation validity.** This is the strongest plausible late-stage loss among the supplied scripts, but the nonoverlapping patches, coefficient defect, BRISQUE-only evaluation, unseeded noise, and fixed BM3D remain.

**Observed result.** No output table or images were supplied.

**Conclusion.** Promising configuration, not a verified result.

**Worth revisiting.** Yes. The optional clean baseline reconstructs this loss with corrected bands.

## E14 Sequential versus parallel operators

**Hypothesis.** Sequential low-, mid-, and high-frequency processing is better than parallel fusion.

**Method.** `sequential_unsupervised_test.py` compares the imported parallel model with a custom sequential Gaussian/Gabor/high-pass model on one salt-and-pepper Lenna image.

**Implementation validity.** Invalid comparison. The models use different kernels, initializations, coefficient constraints, and algebra; only BRISQUE is reported. Sequential coefficients are unconstrained and may extrapolate.

**Observed result.** No table or images were supplied.

**Conclusion.** Inconclusive; it does not isolate sequencing.

**Worth revisiting.** Only as a controlled architecture ablation with matched capacity and constraints.

## E15 Fixed versus trainable kernels

**Hypothesis.** Per-image optimization of initially Gaussian/Gabor kernels improves over fixed deterministic kernels.

**Method.** `trainable_kernels_test.py` makes both kernels arbitrary trainable 5x5 parameters and compares BRISQUE on one Gaussian-noisy Lenna image.

**Implementation validity.** The model is differentiable, but the kernels are unconstrained: the Gaussian need not remain positive or normalized and the Gabor need not remain oriented, band-pass, or zero-mean. It therefore tests a small free convolutional model, not adaptive deterministic filters.

**Observed result.** No result or learned kernels were supplied.

**Conclusion.** Potentially useful capacity test with a mismatched interpretation.

**Worth revisiting.** Yes only with parameterized kernels or explicit physical constraints.

## E16 Classical showdown

**Hypothesis.** The locally optimized method beats mean, median, SVD, bilateral, and BM3D.

**Method.** `ultimate_denoising_showdown.py` evaluates two images under Gaussian and salt-and-pepper noise.

**Implementation validity.** Invalid as a showdown. It reports only BRISQUE despite clean references, uses one parameter setting per classical method, uses nonoverlapping patches, has no seeds, and applies fixed Gaussian-noise BM3D to salt-and-pepper data.

**Observed result.** No table or images were supplied.

**Conclusion.** No competitive claim can be retained.

**Worth revisiting.** Replace with the reproducible protocol in this repository.

## E17 Dense overlap add patch optimization

**Hypothesis.** Fifty-percent-overlapped 64-pixel patches with a Hann window retain local adaptation without seams.

**Method.** `patch_based_comparison.py` reflects the image, optimizes every patch independently, and performs weighted overlap-add using a global selected Gabor kernel.

**Implementation validity.** The stitching method is sound and materially improves E11. However, the optimization is discriminator-only, the underlying coefficient defect remains, Gabor selection still evaluates clipped filter maps as natural images, and only one active Lenna case is run.

**Observed result.** No result images or scores were supplied.

**Conclusion.** Genuinely useful implementation mechanism inside an otherwise inconclusive experiment.

**Worth revisiting.** Yes. A generic, tested overlap-add utility is retained.

## E18 Patch4 comparison duplicate

**Hypothesis.** A 2x2 patch method improves results.

**Method.** `patch4_comparison.py` is a near-duplicate of E17.

**Implementation validity.** The label is wrong: the code still runs dense 64-pixel, 50%-overlapped patches. Only input cropping and output naming differ materially.

**Observed result.** None supplied.

**Conclusion.** Redundant implementation, removed from clean code.

**Worth revisiting.** No separate value beyond E17.

## E19 Clean satellite stress test

**Hypothesis.** The method should not damage an already clean image.

**Method.** `patch4_noise_experiment.py` runs only `deg_type='none'` on one satellite image and compares fixed-noise BM3D with dense patch optimization.

**Implementation validity.** The no-degradation stress test is scientifically valuable. The file name and output label incorrectly call the method 2x2, and BM3D with sigma 0.1 is expected to damage a clean image, so that comparator is unfair.

**Observed result.** No output supplied.

**Conclusion.** Good diagnostic question with an invalid comparator and no evidence.

**Worth revisiting.** Yes; identity preservation should be a mandatory regression test.

## E20 Variants A B and C

**Hypothesis.** Learned edge-enhancement blocks and alternative parallel/cascade layouts improve the simple three-scalar mechanism.

**Method.** `self_fuse_suite.py` introduces random depthwise/pointwise or standard convolution blocks plus mean and median paths.

**Implementation validity.** The base decomposition does not form complementary bands: its sum is not the input. Coefficients are unconstrained, outputs are unclipped during prior evaluation, and the randomly initialized convolutional blocks increase capacity without a controlled comparison.

**Observed result.** `test_suite.py` verifies only shapes and gradient existence. No quality results were supplied.

**Conclusion.** Potentially expressive ideas implemented on a flawed decomposition; inconclusive.

**Worth revisiting.** Only after establishing a correct decomposition and matched-capacity ablation.

## E21 Steerable variants

**Hypothesis.** Oriented high-frequency subbands allow direction-specific denoising.

**Method.** `SelfFuseNet_Steerable` and `SelfFuseNet_HybridSteerable` use four or eight fixed rotated second-derivative-of-Gaussian filters.

**Implementation validity.** Invalid as a steerable-pyramid experiment. The bank is one scale, not a Simoncelli pyramid; it has no analysis/synthesis tight-frame reconstruction; grouped-convolution output ordering is inconsistent with the weight construction and later RGB orientation slices; and the hybrid adds both the isotropic high band and weighted directional responses, double-counting high frequency.

**Observed result.** No metrics or outputs supplied.

**Conclusion.** A good general idea with a bad implementation.

**Worth revisiting.** Yes, but only with a verified analysis/synthesis transform and reconstruction test.

## E22 Broad benchmark of all variants

**Hypothesis.** One of the new variants can outperform simple filters, SVD, BM3D, and the earlier scalar fusion on six degradations.

**Method.** `benchmark.py` evaluates 13 methods on one resized Lenna image using PSNR, SSIM, and MAE.

**Implementation validity.** This is the closest script to a full-reference benchmark, but it remains insufficient: one image, one unseeded draw, untuned baselines, mixed denoising/deblurring tasks, flawed imported models, discriminator parameters left gradient-enabled even though they are not optimized, and no independent hyperparameter selection.

**Observed result.** No `benchmark_results.csv` or output directory was supplied.

**Conclusion.** Useful scaffold, no reliable result.

**Worth revisiting.** The clean benchmark script retains the good parts: seeded degradations, per-image rows, full-reference metrics, and CSV output.

## E23 Metric and packaging smoke tests

`test.py`, `test_pyiqa_lazy.py`, and `zip_code_only.py` are utilities rather than scientific experiments. They confirm intended metric APIs and archive construction, but add no result evidence. They are replaced by explicit optional dependencies, unit tests, and normal project packaging.

## E24 Simoncelli style steerable pyramid coefficient fitting

**Hypothesis.** A multiscale, oriented frequency decomposition gives a visual prior more useful controls than the earlier low/mid/high split.

**Method.** A frequency-domain order-3 angular basis uses four orientations and four smooth log-radial scales. Per crop, NIMA optimizes either 16 bounded gains or 16 gains plus 16 complex soft thresholds. The coarsest residual is retained, and paired analysis/synthesis is checked for identity reconstruction.

**Evaluation.** The exact 100 CBSD68/Kodak24 crops and clipped AWGN realizations from the earlier NIMA/discriminator comparison were reused. The NIMA objective was unchanged. Runtime limited the pilot to 40 optimization updates per variant versus 150 in the earlier low/mid/high run.

**Observed result.** Noisy input: 20.459 dB / 0.3894 SSIM. Gain-only: 20.460 dB / 0.3894. Gain plus threshold: 21.553 dB / 0.4306. Previous NIMA low/mid/high: 24.278 dB / 0.5594. Previous discriminator: 23.252 dB / 0.5206. Maximum identity reconstruction error was 4.77e-7. Crop coordinates and noisy PSNR matched the earlier saved results.

**Conclusion.** The transform is operational, and thresholding improves the noisy input, but this pilot is substantially behind the earlier prior fit. Gain-only stayed effectively at identity. Since this run used fewer updates, the result is preliminary and does not isolate representation quality from optimization convergence.

**Artifacts.** See [the pilot report](simoncelli_pyramid_pilot_2026-10-02.md), the per-crop results in `experiments/coefficient_fitting/results/simoncelli_steerable_100/`, and the interactive `comparison.html` there.

## E25 Matched-step Simoncelli pyramid rerun

**Purpose.** Address E24's unequal optimization duration by rerunning both pyramid variants under the low/mid/high experiment's 150-step schedule.

**Conditions.** Same 100 crops, coordinates, AWGN realizations, sigma, seed, NIMA weights, objective, Adam learning rate 0.01, and batch size 10. NIMA scores the unclipped reconstruction during optimization, as in the low/mid/high run.

**Observed result.** Gain-only: 20.462 dB / 0.3895 SSIM. Gain plus soft thresholds: 20.584 dB / 0.3970. Low/mid/high NIMA: 24.278 dB / 0.5594. Archive discriminator: 23.252 dB / 0.5206. The thresholded pyramid beats the noisy input on 92/100 crops, but trails the previous NIMA run by 3.694 dB on average. The 150-step result is lower than the 40-step pilot's 21.553 dB.

**Conclusion.** Matching the optimization duration leaves a large gap. More steps alone did not improve this method's full-reference scores. Refer to the [matched-run report](simoncelli_pyramid_matched_2026-10-02.md) and the `simoncelli_steerable_100_matched150` results folder.
