# Source Inventory and Evidence Status

## Scope

The supplied material consists of one bachelor thesis, ten Obsidian notes, 29 Python files, and two PyTorch discriminator checkpoints.

No saved experiment CSV, JSON summary, TensorBoard log, training log, Self-Fuse output image, Lenna image, or satellite sample image was present in the supplied archive or recoverable under the searched Downloads paths.

## Primary written sources

| Source | Content used in reconstruction | Evidence status |
|---|---|---|
| `Ավարտական_աշխատանք.docx` | Thesis framing, classical denoising background, CycleGAN rationale, claimed differentiable naturalness contribution, and conclusion | Read structurally: 477 paragraphs, 13 embedded figures, 345 OMML equation tags. Text and every embedded figure were inspected. Direct page rendering was unavailable because the managed runtime did not include LibreOffice. |
| `Self-FuseNet. Data Free Unsupervised Remote Sensing Image Super-Resolution.md` | Low, mid, and high feature extraction and fusion interpretation | Complete note read. Cross-checked against the author-uploaded paper. |
| `VIF ... .md` | VIF/GSM/steerable-pyramid interpretation | Complete note read. |
| `BRISQUE.md` | MSCN, GGD/AGGD, directional products, SVR | Complete note read. |
| `NIQE ... .md` | Patch selection, pristine MVG, Mahalanobis score | Complete note read. |
| `The Steerable Pyramid ... .md` | Radial/angular decomposition and tight-frame intuition | Complete note read. |
| `Bilateral Filter.md` | Edge-aware low-frequency filtering | Complete note read. |
| `Gabor Filter.md` | Oriented localized frequency response | Complete note read. |
| `Generalized Gaussian Distribution ... .md` | GGD moment matching | Complete note read. |
| `Asymmetric Generalized Gaussian Distribution ... .md` | AGGD moment matching | Complete note read. |
| `The Mahalanobis Distance ... .md` | Covariance-aware distance and NIQE intuition | Complete note read. |

The paper itself was cross-checked through its DOI and author-uploaded full text: [Mishra and Hadar, Self-FuseNet, IEEE J-STARS 2023](https://doi.org/10.1109/JSTARS.2023.3239758).

## Code inventory and disposition

| File | Intended role | Clean-project disposition |
|---|---|---|
| `self_fuse_net.py` | Direct deterministic implementation of Self-Fuse algorithms 1-3 | Reimplemented with corrected numeric handling; legacy file not copied. |
| `run_demonstration.py` | Trace deterministic pipeline | Replaced by `scripts/run_baseline.py`. |
| `run_noise_demo.py` | Single-image Gaussian-noise visual demo | History only. |
| `run_denoise_experiment.py` | Save 1,331 fusion candidates and accepted scores | History only; invalid selection/aggregation documented. |
| `scratch_edge_test.py` | Diagnose paper histogram threshold and compare Otsu/mean+std | History only. |
| `scratch_edge_test2.py` | Test bilateral preconditioning before Sobel | Useful idea retained in the stable baseline. |
| `cyclegan_models.py` | ResNet/DAB generators and multiscale PatchGAN discriminator | Only checkpoint-compatible discriminator retained as an optional module. |
| `datasets.py` | Unpaired RealBlur, SIDD, and synthetic degradation dataset | Not retained; dataset construction flaws documented. |
| `train_cyclegan.py` | CycleGAN training | Not retained as a reproducible experiment because no manifest, split, log, or seed is available. |
| `supervised_sanity_check.py` | One-image, three-weight loss sweep with clean target | History only; useful as a representational ceiling diagnostic after redesign. |
| `unsupervised_optimization.py` | Three-band discriminator-only scalar optimization | Reconstructed with corrected complementary bands and identity-preserving weights. |
| `benchmark_experiment.py` | MSE vs SSIM vs adversarial optimization on random degradations | History only. |
| `benchmark_bm3d.py` | Adds fixed-parameter BM3D comparison | History only; comparison is not valid across its mixed degradations. |
| `compare_satelite.py` | Original satellite vs BM3D vs adversarial fusion | History only. |
| `comprehensive_noise_experiment.py` | Whole-image vs 2x2 patch vs BM3D | History only; replaced by generic overlap-add utilities and benchmark protocol. |
| `loss_ablation_study.py` | Adversarial, L1, SSIM, TV loss combinations | History only; TV scaling and evaluation invalidate conclusions. |
| `final_unsupervised_benchmark.py` | Adds mean constraint to 2x2 patch method | Strongest plausible late-stage configuration, reconstructed in optional PyTorch code. |
| `sequential_unsupervised_test.py` | Sequential vs parallel operators | History only; compared incompatible parameterizations. |
| `trainable_kernels_test.py` | Fixed vs unconstrained trainable kernels | History only; trainable kernels cease to represent the claimed deterministic operators. |
| `ultimate_denoising_showdown.py` | Classical methods, BM3D, and patch method | History only; BRISQUE-only, two-image comparison. |
| `patch_based_comparison.py` | 64-pixel, 50% overlap, Hann-windowed local optimization | Overlap-add mechanism retained generically. |
| `patch4_comparison.py` | Near-duplicate of patch-based comparison | Removed as duplicate; its “2x2” label is inconsistent with its dense 64-pixel overlap-add code. |
| `patch4_noise_experiment.py` | Clean satellite stress test | History only; despite its name it runs only `deg_type='none'`. |
| `self_fuse_suite.py` | Variants A/B/C and one-scale “steerable” variants | Not retained; core decomposition and grouped-convolution defects documented. |
| `benchmark.py` | Broad classical and architectural comparison | Replaced by `scripts/run_benchmark.py` and explicit seeded degradations. |
| `test_suite.py` | Gradient/shape smoke test | Replaced by deterministic unit tests. |
| `test.py` | BRISQUE, NIQE, and VIF command-line probe | Replaced by optional lazy metric wrapper. |
| `test_pyiqa_lazy.py` | Dependency smoke test | Removed. |
| `zip_code_only.py` | Archive utility | Removed. |

All 29 Python files parse successfully as Python syntax. That does not make them executable: most require absent image assets, local datasets, nondeclared packages, and hard-coded paths.

## Recoverable binary artifacts

| Artifact | Size | SHA-256 | Status |
|---|---:|---|---|
| `cyclegan_output/resnet/checkpoints/netD_Clean_epoch_50.pth` | 22,264,366 bytes | `30A40FC19BFC023E7D4F004FA567E81F41364FE5EB936F2ECE7E1AA708FEDF7F` | Valid PyTorch zip-format checkpoint container; training quality cannot be inferred. |
| `cyclegan_reproducible_output/resnet/checkpoints/netD_Clean_epoch_100.pth` | 22,264,410 bytes | `709F5C86DA2D7DB46C7FF8224D3A642884937CD8AE115DE1ECEA99186776FDB7` | Valid PyTorch zip-format checkpoint container; most recent recoverable discriminator. |

The clean project does not duplicate these 44 MB of external binary state. The optional loader accepts either checkpoint explicitly and uses the archived discriminator architecture with strict state-dictionary loading.

## Evidence tiers used in the reports

- **Verified fact:** directly present in code, document structure, embedded figures, or recoverable artifact metadata.
- **Recorded claim:** stated in the thesis or comments but not supported by supplied logs/results.
- **Visual observation:** visible in an embedded thesis figure, without a quantitative test.
- **Interpretation:** conclusion from mathematical or implementation audit.
- **Unresolved:** cannot be established from the supplied evidence.
