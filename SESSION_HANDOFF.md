# Exact session handoff

Last durable update: 2026-10-22 Asia/Yerevan. The final holdout is still sealed and
model selection is not locked.

## Completed and verified

- Immutable source-disjoint 20-image development benchmark plus 16 training sources.
- Physical battery watchdog, cumulative active-time budget, atomic optimizer/RNG
  recovery, transient Windows replace recovery, and 25 passing tests.
- Nine classical operators, official KAIR DnCNN/DRUNet, equal controls, frozen DISTS,
  three fixed noise draws, per-source paired intervals, runtime and peak process memory.
- Ineligible pilots and all v3 candidates inspected on all 20 images plus worst panels.
- v4 bank correction (NLM-0.7 selected from the registered sweep), SURE/oracle,
  `v4_lambda_0p0`, and `v4_lambda_0p01` completed, evaluated, and visually inspected.
- Independent reviewer audit addressed: hash-bound lock/holdout, stale-data rejection,
  vendor verification, isolated MUSIQ-to-logit gradient, metric variability, checkpoint
  compatibility, pinned KAIR source commit, provenance, and gate tests.

Substantial run count is 7/10: two ineligible pilots, three v3 runs, and two v4 runs.
The only remaining registered training run is `v4_lambda_0p03`.

Current development result: v4 lambda 0.01 is +0.00239 dB PSNR over CBM3D with a
positive paired interval and improves DISTS, but has a tiny consistent SSIM regression.
It is about 0.41 dB behind DnCNN and 1.03 dB behind DRUNet. This is not yet evidence
for a classical improvement claim, and neural competitiveness is already unsupported.

## Exact resume sequence

From the repository root, first inspect `outputs/research_v3/watchdog.json`,
`session_budget.json`, and `run_budget.json`. Stop if battery is 15% or lower. Then:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.03
.venv\Scripts\python.exe scripts\research_session_v3.py evaluate --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py views --method v4_lambda_0p03 --split benchmark
```

Open all four v4-lambda-0.03 pages and its worst panel, add the observation to
`experiments.csv`, and rerun the 25-test suite. Do not run another architecture:
A2-A4 gates are closed. Create `visual_audit.json` listing every inspected candidate,
then run `lock`. Only after the lock succeeds may `holdout` be run once.

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py lock
.venv\Scripts\python.exe scripts\research_session_v3.py holdout
```

After holdout, inspect all four final pages and the worst panel. Package the selected
`deploy.pt`, model JSON, lock, summaries, manifests, environment/provenance records,
and pinned small KAIR sources into a tracked `research_artifacts/final/` directory.
Update the final report, tests, `experiments.csv`, `next_steps.md`, and commit them.

Do not rerun `prepare`, reset the budget, tune from holdout, or claim full-image
Urban100/CBSD68/Kodak results. All generated arrays/checkpoints remain under ignored
`outputs/research_v3/` and are protected by hashes; the tracked protocol and this
handoff are the durable map.
