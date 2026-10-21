# How to inspect and resume the experiments

This is the operator's guide for checking the research if the automated session is
temporarily interrupted. Run commands from the repository root. Use the project
environment at `.venv\Scripts\python.exe`; do not recreate the benchmark or delete
partial caches.

## 1. Read the frozen protocol first

Read these files in order:

1. `research_plan.md` — scientific protocol, claims, metrics, and cumulative budget.
2. `architecture_registry.md` — every scheduled and gated architecture, with exact
   layers, parameters, objectives, and expansion criteria.
3. `benchmark20.json` — immutable 20-image development set and disjoint training set.
4. `experiments.csv` — chronological run/failure/inspection ledger.
5. `next_steps.md` — current cache-aware resume sequence.
6. `MODEL_AND_METRIC_PROVENANCE.md` — exact MUSIQ, DISTS, neural-baseline, and
   environment provenance, including preprocessing and domain-shift cautions.

The archived 128x128 CBSD/Kodak crops are development data, not a fresh test set.
The final Urban100 081-100 holdout must remain absent until `locked_model.json` exists.

## 2. Confirm safety before any expensive command

Open these JSON files:

- `outputs/research_v3/watchdog.json`: current or last process ID, power source,
  battery percent, stop flag, and stop reason.
- `outputs/research_v3/session_budget.json`: original segment, carried active seconds,
  equivalent resumed start, and hard deadline.
- `outputs/research_v3/run_budget.json`: every started substantial training run.
- `outputs/research_v3/safety_verification.json`: low-battery and exact recovery test.

If battery is 15% or lower, do not start or resume an experiment. A stale watchdog
timestamp means no research process is currently proving liveness; it is not evidence
that a process is still running. Before resuming after an interruption, run:

```powershell
.venv\Scripts\python.exe scripts\verify_session_safety.py
```

That check must report exact uninterrupted-versus-resumed parameter equality and a
frozen MUSIQ scorer. It is a smoke test, not a substantial training run.

## 3. Understand the output tree

```text
outputs/research_v3/
  benchmark_data.npz       frozen clean + 3 noisy development realizations
  training_data.npz        frozen clean + noisy training sources (clean is audit-only)
  cache/benchmark/         one exact output tensor and timing tensor per method
  cache/training/          operator outputs used by MC-SURE
  runs/<run_id>/           model.json, deploy.pt, checkpoint.pt, calibration
  metrics/benchmark/       all per-image/per-seed metric rows, one JSON per method
  views/benchmark/<method>/page_1.png ... page_4.png and worst_XX.png
  benchmark_summary.json   mean PSNR, SSIM, DISTS, runtime, peak process memory
  benchmark_results.csv    every individual metric observation
  benchmark_paired_deltas.json  paired source-cluster bootstrap intervals
  oracle_v4.json           development-only diagnostic; never a deployable result
  locked_model.json        immutable model selection and hashes, when selection ends
  holdout_*                must appear only after the lock
```

An `.npz` file ending in `_partial` is a recovery checkpoint, not a finished result.
The corresponding file without `_partial` is authoritative. The runner verifies the
hashes of the frozen clean/noisy arrays at every resumed development command.

## 4. Review numerical results correctly

Open `outputs/research_v3/benchmark_summary.json` for the compact table, then verify
interesting cases in `benchmark_results.csv`. Interpret metrics as follows:

- PSNR: higher is better and is the primary selection metric.
- RGB SSIM: higher is better.
- DISTS: lower is better. It is independent and never trains or selects weights.
- Runtime is wall time per image from the cached inference run.
- Memory is whole-process peak working set, not model-exclusive memory.

Use `benchmark_paired_deltas.json` for claims, not just mean differences. Its unit is
one source image after averaging the three paired noise draws. An improvement over
the strongest classical method requires a positive 95% interval and no consistent
SSIM/DISTS regression. “Neural competitive” requires being within 0.2 dB without a
perceptual decline. The reference oracle is only a complementarity diagnostic.

## 5. Inspect every saved image, not a cherry-picked example

For each meaningful candidate, generate the fixed comparison sheets if absent:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py views --method RUN_ID --split benchmark
```

Open all five files under `outputs/research_v3/views/benchmark/RUN_ID/`:

- `page_1.png` through `page_4.png`: all 20 images, five rows per page.
- `worst_XX.png`: candidate's worst MSE delta relative to color BM3D, plus 5x error.

Each row is clean reference, noisy input, candidate, color BM3D, and DRUNet. Inspect
at native scale and record in `experiments.csv`:

- residual chroma/luminance noise;
- texture or thin-line erasure;
- edge halos/ringing;
- color or brightness shifts;
- repeated-pattern instability;
- whether an apparent perceptual gain is merely extra smoothing.

Do not choose a next run until the candidate's four pages and worst panel have been
opened and the finding is written to the ledger. The clean images are used only for
development evaluation and the diagnostic oracle, never for the training loss.

## 6. Check a learned model and its recovery state

For each folder in `outputs/research_v3/runs/`, inspect:

- `model.json`: final convex weights, lambda, frozen-MUSIQ flag, preprocessing,
  input-gradient norm, checkpoint hash, and full learning history.
- `deploy.pt`: minimal reproducible bank/weights/config payload.
- `checkpoint.pt`: logits, Adam state, Python/NumPy/Torch RNG states, next step,
  history, and explicit `scheduler=None`, `scaler=None`.
- `quality_calibration.json`: training-only MUSIQ mean and standard deviation.

The current run IDs are `v4_lambda_0p0`, `v4_lambda_0p01`, and `v4_lambda_0p03`.
The first is an exact MC-SURE control. The other two are the only predeclared learned
MUSIQ ablations for the NLM-selected bank. The completed v3 triplet is preserved as
the controlled predecessor. Two still older pilot runs are ineligible. Together the
two pilots + v3 triplet + v4 triplet consume 8 of 10 allowed runs.

## 7. Cache-aware resume sequence

Commands are idempotent: finished output files are reused, and partial arrays or
training checkpoints resume. Do not run `prepare`; it intentionally refuses to
overwrite the benchmark.

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py acquire
.venv\Scripts\python.exe scripts\research_session_v3.py classical --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py neural --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py classical --split training
.venv\Scripts\python.exe scripts\research_session_v3.py sure
.venv\Scripts\python.exe scripts\research_session_v3.py oracle
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.01
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.03
.venv\Scripts\python.exe scripts\research_session_v3.py evaluate --split benchmark
```

After each run, use the `views` command and update `experiments.csv`. A completed
file is skipped; a checkpointed positive-lambda run restores the real Adam and RNG
state. Never change lambda, bank order, noise draws, or benchmark after seeing results.

## 8. Model-selection and holdout gate

The higher-capacity A2-A4 designs in `architecture_registry.md` are closed unless
their preregistered development gates are met. As of the completed oracle diagnosis,
global fusion adds only 0.0068 dB over CBM3D and per-image adaptation adds only
0.0316 dB over global fusion, so those gates are currently closed.

Only after all three A1 variants, all comparators, all-20 visual inspections, and the
ledger are complete may model selection be frozen:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py lock
```

Inspect `locked_model.json` and verify its benchmark, config, and script hashes. Then,
and only then, run the final holdout once:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py holdout
```

Open all four holdout pages and the worst panel, then report the holdout results as
Urban100 081-100 center 128 crops—not full Urban100 or standard full-image scores.
The holdout cannot justify changing the architecture or selecting another run.

## 9. If something fails

Preserve the traceback, partial cache, checkpoint, and `watchdog.json`. Add one row to
`experiments.csv` with the failure, cause, evidence-based revision, controlled retest,
and whether it consumed a substantial run. Do not delete a failed run or reset the
run/time ledger. Exact current commands and any interruption-specific note belong in
`next_steps.md`.
