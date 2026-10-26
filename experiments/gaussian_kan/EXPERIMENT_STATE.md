# Experiment state — 2026-10-27

## Current position

- Milestones 0 and 1 are complete. Milestone 2, the 1,000-step primary KAN run, is active and intentionally interrupted at step 31 after a bounded pilot.
- Active experiment: `kan_primary_seed_1729`. Completed experiments: none. Pending: primary KAN, matched MLP, affine model, Gaussian parameter estimate.
- Latest resumable checkpoint: `checkpoints/latest.pt`, step 31. Preserved pilot snapshot: `checkpoints/pilot_step_000031.pt`, SHA-256 `8547302FFEC2B24F6634B6241A23FFD6F4B9A21E1A31282A970E1558AD9939A9`. Checkpoints are ignored by Git and must remain accessible locally.
- Latest committed milestone: Milestone 0 at `b7eb1c1`; receipt commit `1d78a9a`. Both pushed to `origin/main`. Milestone 1 commit and push are pending as this entry is written.

## Attempted and completed

- Milestone 0 established the isolated study, configuration, atomic JSON state, atomic PyTorch checkpoint IO, status command, ignore rules, and resume instructions. Config and checkpoint round-trip validation passed.
- Milestone 1 used the genuine pure-PyTorch B-spline KAN from `shawcharles/efficient-kan` commit `0240ea1`, widths `[1,8,8,1]`, 32 intervals, order 3, 2,960 trainable parameters, CPU. The pinned package was added to the existing research `.venv` after locating it in the earlier project chat.
- Ran 20 steps with independent batches of 2,048 uniform and Gaussian samples, then restored the saved model, optimizer, and RNG states and ran steps 21–30. A one-step additional resume reached step 31.
- Exact replay of steps 21–30 from the step-20 checkpoint reproduced every saved model tensor and PyTorch RNG state at step 30. Initial-to-step-20 and step-20-to-step-30 parameters both changed. Optimizer state contains nine parameter entries.
- Fixed validation squared W2 fell from `0.0357447` initially to `0.0200024` at step 20, `0.0127736` at step 30, and `0.0121833` at step 31. At step 31 output mean was `-0.01955` and output standard deviation `0.10792` versus target `0.2`. This is a functioning pilot, not a learned Gaussian distribution yet.
- Files created: `models.py`, resumable `train.py`, `visualize.py`, `tests/test_resume.py`, metric CSV, pilot validation JSON, and `results/figures/pilot_initial_and_step30.png`. Visual inspection shows the distribution broadening, but the pilot map is still nonmonotone and far from the increasing inverse CDF.

## Errors and limits

- Initial Windows checkpoint fsync used a read-only handle and failed; corrected to a writable handle and round-trip validation passed.
- The first pilot logger wrote duplicate rows when the bounded session ended exactly at a validation step. The condition was corrected; existing duplicates were removed; a step-31 resume confirmed unique metric steps `[0,20,25,30,31]`.
- The primary result, baseline comparisons, finite-sample reference, full function and tail evaluation, additional seeds, and capacity tests remain undone. No final scientific claim has been made.
- The older checkout at `C:\Users\saroy\Codes\New-Denoising-Algorithm` is behind the active project checkout. The active project environment is `C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv` (Python 3.12.14, PyTorch 2.14.1+cpu, CUDA unavailable).

## Next concrete action

From this directory in PowerShell, continue the same run in bounded sessions:

```powershell
& 'C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv\Scripts\python.exe' train.py --resume latest --max-runtime-seconds 600
```

Inspect `train.py --status`, the metric CSV, and the latest checkpoint after each session. Do not start without `--resume` while this run is interrupted. After step 1,000, verify final and best checkpoints and commit the Milestone 2 evidence.
