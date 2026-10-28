# Learning a Gaussian generator with a spline KAN

**Completed study — 29 October 2026**

The KAN learned an approximately Gaussian output law from independent, unpaired uniform and Gaussian samples. It learned a **decreasing** quantile-like transformation. Central accuracy was good, while extreme tails and optimization fluctuations remained measurable. The tested KAN outperformed the parameter-matched MLP under this protocol, but Gaussian parameter estimation was substantially cheaper and more accurate when its distributional assumption was available.

## Experimental setup

The primary generator was a genuine cubic B-spline KAN with widths [1,8,8,1], 32 grid intervals, and 2,960 trainable parameters. It used the pure-PyTorch [efficient-kan implementation pinned to commit 0240ea1](https://github.com/shawcharles/efficient-kan/tree/0240ea138fe257d800c9bfa561025e1c44aedc71). All layers used the implementation's [-1,1] knot range; no adaptive grid updates or symbolic branch were used. Training used float32 on CPU, Adam at 0.001, 2,048 samples per batch, and 1,000 optimization steps.

Each step drew fresh independent Z ~ U(0,1) and N ~ Normal(0,0.2²) sample sets. Numerical endpoint values were excluded at float32 precision. The loss was the mean squared difference between **independently sorted** generated and target samples. The analytical inverse CDF, Gaussian density, and target parameters were absent from generator construction and forward inputs. The controlled target sampler used the known scale to supply observations.

Fixed validation used 8,192 independent latent/target draws with separate seeds and never consumed the training random stream. It selected checkpoints before independent testing. The first test used 100,000 samples with latent seed 824501 and target seed 912877. The later capacity confirmation used another 100,000-sample pair, seeds 20261029 and 20261030. Gaussian-versus-Gaussian sampling references each used ten independent pairs of equal size.

The MLP had widths 1→52→52→1, SiLU activations, and 2,913 parameters. The affine generator had two parameters. Both used the same Adam, sample, step, and validation protocol. A supplementary closed-form affine fit used sorted independent samples to separate affine capacity from slow Adam convergence. Gaussian mean/std estimation used 2,048,000 independent target draws and the Gaussian family assumption; the true inverse-CDF oracle was evaluation only.

The neural models shared a seed label and sampling protocol, not identical training draws: different architectures consume different random numbers during initialization. Validation and test draws were exactly shared across comparators.

## Completed runs and recovery

Seven optimization runs completed: five KANs (three primary seeds, one smaller KAN, one finer-grid KAN), one MLP, and one affine generator. Two supplementary closed-form fits were evaluated, together with the true oracle. No planned training run remains incomplete. The primary pilot and later main training were one continued trajectory, not separate research runs.

The pilot stopped at step 20, resumed to step 30, then step 31, and continued to step 1,000. Exact replay from step 20 reproduced step-30 model tensors and PyTorch random state. The configurable runner separately reproduced uninterrupted versus resumed weights, Adam state, and random state, and rejected silent overwrites or changed configurations on resume.

All runs preserved best and final weights, metrics, configuration, optimizer state, random state, and periodic 100-step checkpoints. The local checkpoint manifest records **215 files, 10,600,937 bytes**. Their SHA-256 hashes can be verified with scripts/build_checkpoint_manifest.py --verify. Checkpoints are ignored by Git and included in the portable deliverable bundle; a Git-only checkout contains measurements and figures but needs those artifacts to inspect saved weights.

## Independent comparison

| Generator | First-test W2 | Mean error | Std error | Parameters |
| --- | ---: | ---: | ---: | ---: |
| KAN selected at step 400 | 0.00776 | +0.00431 | −0.00067 | 2,960 |
| KAN final at step 1,000 | 0.00748 | −0.00631 | −0.00144 | 2,960 |
| MLP selected at step 900 | 0.04294 | +0.00322 | −0.00650 | 2,913 |
| Affine Adam, step 1,000 | 0.04765 | −0.00075 | −0.02457 | 2 |
| Empirical quantile-fitted affine | 0.04300 | −0.00047 | −0.00435 | 2 |
| Gaussian parameter fit | 0.00165 | −0.00054 | +0.00005 | 2 estimated |
| True inverse-CDF oracle | 0.00166 | −0.00042 | +0.00003 | 0 |

The first test's Gaussian sampling reference averaged W2 **0.00155**, SD **0.00034**. Thus the KAN learned a useful approximation, with remaining error above sampling variation. The Gaussian fit and oracle were near that reference.

The final KAN happened to score slightly better than the validation-selected checkpoint on the first test; on the later confirmation pair the ordering reversed. Both are retained and reported. No test-based checkpoint replacement was made. This sensitivity favors averaging validation over larger or multiple samples.

## Transformation, tails, and capacity

The selected primary KAN was strictly decreasing on a 10,000-point grid over [1e-4,1-1e-4]. Its RMSE against -0.2 Φ⁻¹(z) was **0.00642**, versus **0.39905** against the increasing reference. Distribution matching does not determine orientation or uniquely recover an original stochastic mechanism.

At the 0.1% and 99.9% quantiles, its first-test signed errors were **+0.06055** and **−0.04530**. It produced no absolute values above 0.6 in that test, compared with target probability **0.00270**. The final checkpoint reached a fraction **0.00193**. A finite ordinary KAN does not exactly represent the unbounded inverse CDF at the endpoints.

| KAN configuration | Parameters | Confirmation W2, validation-selected |
| --- | ---: | ---: |
| Primary grid 32, seed 1729 | 2,960 | 0.00624 |
| Primary grid 32, seed 1730 | 2,960 | 0.00602 |
| Primary grid 32, seed 1731 | 2,960 | 0.00527 |
| Smaller [1,8,1], grid 32, seed 1729 | 592 | 0.00663 |
| Primary grid 64, seed 1729 | 5,520 | 0.00577 |

Across the three primary seeds, confirmation W2 averaged **0.00584**, sample SD **0.00051**. All selected maps were strictly decreasing on the measured grid. The 592-parameter model, one fifth of the primary parameter count, retained similar central accuracy. It is the smallest tested configuration; the true minimum is unknown.

Grid 64 improved seed-1729 extreme quantile errors and produced more tail mass, but did not remove the limitation or reach the sampling reference. Seed variation was comparable to the gain in overall W2. These observations did not justify a larger/deeper search.

## Cost and interpretation

The first primary training loop took 108.90 seconds, versus 3.36 seconds for the MLP and 2.90 seconds for affine Adam. Gaussian parameter fitting took about 0.045 seconds. Repeat primary KANs in the later session took about 31–32 seconds; the smaller and finer-grid KANs took 18.55 and 54.42 seconds. These are recorded CPU observations with four PyTorch threads, not controlled cross-session performance benchmarks.

For a known Gaussian law, the KAN's computational cost is not justified against two-parameter estimation. Its value here is evidence that spline edge functions can learn a nonlinear distribution transform without receiving the target family. The MLP's nearly affine result is preserved under the tested budget; a different activation, initialization, learning-rate schedule, or training duration could change that comparison.

The selected architecture was expressive enough for the center of the distribution. Tail errors and stochastic optimization remain. Three seeds provide preliminary reproducibility evidence, not a broad stability guarantee. The architecture alternatives and MLP have one seed each.

## Answers to the research questions

| Question | Measured answer |
| --- | --- |
| Did KAN learn the Gaussian output law? | Approximately; much closer than this MLP/affine protocol, still above the sampling reference and imperfect in the tails. |
| Which transformation? | A strictly decreasing quantile-like map on the evaluated interior. |
| Did it approximate the theoretical inverse CDF? | It approximated the decreasing equivalent, with interior RMSE 0.00642 for the primary selected checkpoint. |
| Was primary capacity sufficient? | Sufficient for good central accuracy; exact global tail representation is impossible with this finite bounded-input model. |
| How much capacity was needed? | A tested 592-parameter KAN retained similar central accuracy; minimum necessary capacity was not identified. |
| Better than MLP/statistical baselines? | Better than the tested MLP and affine models; worse than Gaussian parameter estimation and the oracle. |
| Was cost justified? | Not for known Gaussian generation. Potential usefulness for unknown non-Gaussian laws remains a research question. |
| Stable across seeds? | Three primary seeds gave confirmation W2 0.00527–0.00624; tails and checkpoint ranking varied. |
| Worth further investigation? | Yes for controlled unpaired scalar distribution learning; this result alone does not justify claims about noisy-image mechanism recovery. |

## Recommended next experiment

First improve validation averaging and predeclare a small learning-rate/tail sensitivity study. Then compare the smaller KAN, an improved MLP control, empirical quantile interpolation, and appropriate statistical baselines on skewed and mixture scalar targets using unpaired samples. Keep held-out evaluation separate from architecture selection and report all seeds.

Before moving to corrupted images without clean references, specify assumptions linking content and noise (for example conditional independence, repeated observations, or a restricted noise family). Matching noisy-image marginals alone does not identify their decomposition.

## Notebook and reproducibility

Open Gaussian_KAN_Lab.ipynb beside these files. It contains 32 cells, including 16 executed code cells with saved outputs. It checks samples, sorted loss, spline gradients, exact recovery, checkpoint hashes, trajectories, baseline results, quantiles, tails, seeds, and capacity. Its optional experiment section offers KAN resolution/widths, SiLU or tanh MLPs, affine models, and an explicitly exploratory W1 objective. New runs use isolated directories and explicit resume.

From the repository root on this machine:

~~~powershell
.\.venv\Scripts\python.exe -m jupyterlab experiments\gaussian_kan\Gaussian_KAN_Lab.ipynb
~~~

For a standalone bundle, install requirements-notebook.txt in a compatible environment, open this directory in Jupyter, and select that interpreter. The HTML export provides a static reading copy. RESUME_INSTRUCTIONS.md, EXPERIMENT_STATE.md, environment_study.json, and the metric CSV/JSON files preserve all essential state outside the conversation.
