# Exact session handoff

Last durable update: 2026-10-22 Asia/Yerevan. Development is complete through the
eighth and final substantial training run. The final holdout remains sealed until the
model lock is written.

## Completed and verified

- Fixed, source-disjoint 20-image development benchmark, 16 training sources, and
  three deterministic unclipped AWGN draws at sigma 25/255.
- Battery watchdog, cumulative four-hour budget, atomic optimizer/RNG recovery,
  transient Windows-file recovery, and 25 passing tests.
- Nine classical operators; official pinned KAIR DnCNN and DRUNet; frozen DISTS;
  source-cluster paired intervals; runtime and peak-process-memory accounting.
- Frozen pretrained MUSIQ/KonIQ transformer with verified checkpoint, isolated
  nonzero logit gradient, standardized IQA term, and MC-SURE fidelity objective.
- Two documented ineligible pilots, three v3 runs, and the evidence-based v4 NLM-0.7
  triplet. All meaningful candidate pages (20 images each) and worst panels were
  personally opened; see `visual_audit.json` and `experiments.csv`.
- Independent literature research and independent protocol/code review incorporated.

Run count is 8/10: two preliminary pilots, three v3 runs, and three v4 runs. No more
training is justified or allowed in this bounded session. Spatial, multiscale, and
sequential gates are closed by oracle evidence.

## Locked selection decision to execute

Select `v4_lambda_0p01`, the highest-development-PSNR positive-lambda candidate.
Development: PSNR 31.54778 dB (+0.00239 vs CBM3D, paired 95% CI +0.00148 to
+0.00317), DISTS improves by 0.000196, and SSIM regresses by 0.0000455 (CI
-0.0000719 to -0.0000232). It remains 0.41 dB behind DnCNN and 1.03 dB behind
DRUNet. Therefore it does not yet support a full improvement claim.

The SURE-only v4 control reaches 31.55998 dB but also has a larger consistent SSIM
regression. The final v4 lambda 0.03 run reaches 31.54686 dB and is rejected by the
registered selection rule.

## Exact resume sequence

From the repository root, inspect `outputs/research_v3/watchdog.json`,
`session_budget.json`, and `run_budget.json`; stop if battery is 15% or lower. Then:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\research_session_v3.py lock
.venv\Scripts\python.exe scripts\research_session_v3.py holdout
```

The `holdout` command is one-shot and resumable. Never modify code/config/model after
`lock`, never tune from holdout, and never rerun `prepare` or reset the cumulative
budget. After holdout, open all four selected pages and its worst panel, plus DnCNN
pages if generated; record observations without changing selection. Then update the
final report and package the selected deploy/model/lock, summaries, manifests,
provenance, environment record, small pinned vendor sources, and representative view
images into Git. Do not commit large pretrained checkpoints or data arrays.

Generated arrays/checkpoints are ignored under `outputs/research_v3/` but are bound by
SHA-256 manifests. The tracked protocol, architecture registry, audit, experiment log,
and this file are the durable recovery map.
