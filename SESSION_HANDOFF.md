# Exact session handoff

Last durable update: 2026-10-22 Asia/Yerevan. The bounded research session and sealed
holdout are complete. No training or model-selection work remains.

## Final state

- Selected run: `v4_lambda_0p01`.
- Architecture: global convex fusion of identity, bilateral, NLM-0.7, and CBM3D.
- Weights: `[0.0000550514, 0.0019913730, 0.0035475527, 0.9944059849]`.
- Training count: 8/10 substantial runs; no capacity-expansion gate opened.
- Holdout: Urban100 081-100 center 128 crops, three fixed noise draws, completed once
  after a SHA-256-bound lock. `holdout_complete.json` binds the lock and frozen data.
- Safety: physical watchdog and exact checkpoint recovery verified; last expensive
  stage completed above the 15% threshold.
- Tests: 25 passed before lock; run them again after checkout.
- Visual audit: every meaningful development candidate, final selected output, DnCNN,
  and equal control were opened on all 20 relevant images plus worst panels.

The final claim is negative. Holdout fusion-versus-CBM3D is +0.00293 dB PSNR with
positive paired interval and +0.000099 DISTS improvement, but SSIM regresses by
0.0000483 with a wholly negative interval. DRUNet is 1.499 dB better. See
`FINAL_REPORT.md` for all metrics, limitations, and literature.

## Fresh-checkout reproduction

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-research-lock.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\apply_locked_fusion.py input.png output.png
```

The selected model JSON and `deploy.pt`, lock, summaries, paired deltas, per-image
results, final views, and pinned small KAIR sources are force-tracked under their
original `outputs/research_v3/` paths. Large data arrays, source images, and pretrained
checkpoints remain intentionally untracked; their URLs and SHA-256 hashes are recorded
in `MODEL_AND_METRIC_PROVENANCE.md` and the lock. Consequently, a fresh clone can audit
the tracked result rows and deploy the selected model, but it cannot recompute the
completed holdout until those inputs are reacquired. The tracked completion marker is
deliberately rejected when its bound `holdout_data.npz` is absent.

To audit rather than rerun, start with `FINAL_REPORT.md`, `visual_audit.json`,
`experiments.csv`, `architecture_registry.md`, and `ARTIFACT_MANIFEST.md`. Never use
the completed holdout to choose a new architecture while presenting it as untouched.

If extending this research in a new session, create a new protocol version and a new
unseen holdout. The strongest evidence-based next idea is not additional gating
capacity over the same bank; it is a genuinely more complementary operator bank.
