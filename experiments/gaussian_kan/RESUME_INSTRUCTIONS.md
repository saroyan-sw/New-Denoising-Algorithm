# Resume instructions

1. Read `EXPERIMENT_PLAN.md`, `EXPERIMENT_STATE.md`, and `experiment_state.json`.
2. Inspect `git status --short --branch`, `git log -5 --oneline`, and `git remote -v` from the repository root.
3. Install `requirements.txt` in a Python environment with compatible PyTorch. On the current Windows machine, the existing research `.venv` has PyTorch and scientific packages. Install the pinned `efficient-kan` dependency there or into an isolated local package directory and set `PYTHONPATH` accordingly.
4. Run `python train.py --status` in this directory. Check that `latest_checkpoint` exists and is loadable before resuming. Ignored checkpoints need separate local preservation.
5. Continue the first unfinished milestone. To resume training, use `python train.py --resume latest --max-runtime-seconds 600` or `python train.py --resume latest --max-steps-per-session 200`. Never start an interrupted run from step 0 without an explicit deliberate reset.
6. Update both state files after verified work. Commit and push each validated milestone. Record push failures honestly.

For the current Windows checkout, the exact next command from this directory is:

```powershell
& 'C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv\Scripts\python.exe' train.py --resume latest --max-runtime-seconds 600
```

The step-31 checkpoint is local and ignored by Git. Preserve or copy `checkpoints/latest.pt` and the other checkpoint files when moving the study; a fresh remote checkout alone cannot resume the interrupted optimizer trajectory.

The existing project virtual environment used in earlier sessions is `C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv`. Its path is machine-specific and is not required by the project code.
