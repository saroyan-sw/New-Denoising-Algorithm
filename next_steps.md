# Resume — session in progress

For a complete inspection map and failure-safe operating procedure, read
`SESSION_HANDOFF.md` and `EXPERIMENT_REVIEW_GUIDE.md` first.

Current checkpoint: v4 lambda 0 and 0.01 are complete/evaluated/all-20 inspected;
v4 lambda 0.03 has not started. Run count is 7/10. Model lock is absent and final
holdout is unopened. The next allowed training command is exactly:

```powershell
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.03
```

Working directory: this repository. Python: `.venv/Scripts/python.exe`.
The budget remains 10 substantial runs or four cumulative active hours. The original
segment consumed 2,208.606 seconds. Following the user's explicit 2026-10-22 resume,
the watchdog uses equivalent start Unix 1792610865.57409, leaving 12,191.394 seconds
at the resume instant. Never reset it automatically again. Two preliminary runs count.
All scheduled and gated architectures are frozen in `architecture_registry.md`.

The initial 20-crop protocol incorrectly included fitting images and has been superseded
by benchmark20.json with 16 separate training sources. Preserve the old artifacts.
Protocol v4 uses three benchmark noise draws, the development-selected NLM-0.7 compact
global convex bank, frozen MUSIQ, MC-SURE fidelity, independent DISTS, CBM3D/NLM and
DnCNN/DRUNet. The v3 NLM-0.9 triplet remains as controlled predecessor evidence.

Commands (run from repository root; do not recreate the frozen benchmark):

```powershell
.venv\Scripts\python.exe scripts\verify_session_safety.py
.venv\Scripts\python.exe scripts\research_session_v3.py acquire
.venv\Scripts\python.exe scripts\research_session_v3.py classical --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py neural --split benchmark
.venv\Scripts\python.exe scripts\research_session_v3.py classical --split training
.venv\Scripts\python.exe scripts\research_session_v3.py sure
.venv\Scripts\python.exe scripts\research_session_v3.py oracle
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.01
.venv\Scripts\python.exe scripts\research_session_v3.py train --lambda-iqa 0.03
.venv\Scripts\python.exe scripts\research_session_v3.py evaluate
```

After each candidate: generate/open all four pages via `views --method RUN_ID`, inspect
the worst native-resolution panel, and record observations. Only after every comparison
and visual audit is complete may `lock` be called. `holdout` then resumes cached per-image
baseline inference and performs final evaluation. Do not develop after the lock.

Current limitations: archived 128px crops and historically exposed source images;
category labels from statistical proxies need visual qualification; no full-dataset claim.
