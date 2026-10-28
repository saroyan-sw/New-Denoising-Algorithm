# Experiment state — 2026-10-29

## Current position

- Milestones 0–6 are complete. All eight planned experiment IDs are completed; no training run is active, failed, or interrupted.
- The final report, executed notebook, checkpoint manifest, and portable bundle are verified.
- Latest pushed result milestone: Milestone 5 at e540bda, receipt 3c1a50d. Milestone 6 is verified and awaits its final commit/push receipt.
- Primary final: checkpoints/latest.pt, step 1,000; validation-selected primary: checkpoints/best_step_000400.pt.
- Other weights: checkpoints/mlp/, checkpoints/affine/, and runs/<run_id>/checkpoints/.
- The 215 checkpoint hashes are recorded in CHECKPOINT_MANIFEST.json. Model binaries are ignored by Git and included in outputs/Gaussian_KAN_Study_Bundle.zip.

## What was completed and actually worked

0. Located the existing project environment via the earlier project chat: Python 3.12.14, PyTorch 2.14.1+cpu, CUDA unavailable. Established state/configuration/ignore rules, atomic checkpoint IO, and Git milestones.
1. Verified genuine cubic spline KAN forward propagation, finite gradients, parameter updates, a 31-step pilot, and exact step-20-to-step-30 replay of weights and PyTorch RNG state.
2. Continued that same trajectory to 1,000 steps. Verified ten periodic checkpoints plus best and final validation measurements. Best step: 400.
3. Completed the parameter-matched MLP and affine Adam baselines, empirical sorted-quantile affine fit, Gaussian parameter fit, and inverse-CDF oracle comparison.
4. Evaluated fresh 100,000-sample output laws, ten Gaussian sampling-reference pairs, moments, quantiles, monotonicity, interior function error, and tails. Six scientific figures were inspected.
5. Completed two more primary seeds, smaller32, and primary64 under the fixed queue. Verified 40 new periodic checkpoints and selected-checkpoint metrics. Evaluated a fresh confirmation pair and inspected both capacity figures.
6. Created FINAL_RESEARCH_REPORT.md, NOTEBOOK_GUIDE.md, ARTIFACTS.md, environment/requirements records, lab comparison helpers, and Gaussian_KAN_Lab.ipynb. Its 32 cells include 16 executed code cells, ten embedded figures, and zero errors. Default execution ran exact recovery, gradients, live comparison, and artifact checks. Separate integration checks validated MLP/tanh and KAN/W1 optional experiments, explicit resume, and rejection of mismatched target comparisons. The portable ZIP passed CRC, all checkpoint hashes, and exact notebook-byte verification.

## Measured findings

First independent-test W2: KAN selected 0.007763; KAN final 0.007481; MLP 0.042940; Gaussian parameter fit 0.001646; oracle 0.001658. Gaussian sampling-reference mean 0.001555, SD 0.000336.

The selected KAN was strictly decreasing over the measured interior grid, with RMSE 0.006421 against the decreasing quantile reference. Extreme tails remained clipped. The first-test final/best ranking flipped on the later confirmation pair; both are reported without test-based reselection.

Confirmation W2 for primary seeds 1729/1730/1731: 0.006242/0.006020/0.005266 (mean 0.005843, SD 0.000512). The 592-parameter smaller KAN reached 0.006627; grid64 with 5,520 parameters reached 0.005765. Finer resolution helped some tail errors. Minimum necessary capacity remains unknown.

## Encountered errors and corrections

- Initial inspection looked at the generated workspace and an older repository checkout. The active .venv was found in the prior project chat and reused.
- Windows fsync failed on a read-only checkpoint handle; changed to a writable handle and verified round-trip IO.
- Pilot metrics duplicated session-end validation rows; corrected the condition and verified unique steps.
- Evaluation consistency exposed float32 subtraction rounding in affine std error; arrays now use float64 for evaluation and the same frozen samples were rechecked.
- Initial notebook execution tried a read-only default IPython history directory; the harness now uses project-local caches.
- Git pushes from the older isolated clone needed a one-command safe-directory setting. All milestone pushes succeeded; no force-push was used.

## Artifact location and continuation

The authoritative study is now in the active project checkout under New-Denoising-Algorithm/experiments/gaussian_kan. It was copied from the earlier output checkout on 2026-10-29 after a Git fast-forward; all 106 then-existing checkpoint hashes matched. No synced file under sources/ was changed.

Exact next command from this directory:

~~~powershell
& '..\..\.venv\Scripts\python.exe' -m jupyterlab Gaussian_KAN_Lab.ipynb
~~~

No required scientific work remains. Optional experiments use independent named directories under explorations/, with explicit resume. Do not repeat completed research runs without documenting a deliberate rerun. Read the final report for limitations and the recommended next experiment.
