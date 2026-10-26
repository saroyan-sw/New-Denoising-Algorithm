# Experiment state — 2026-10-27

## Current position

- Milestones 0–2 are complete. Milestone 3, baseline comparisons, is next.
- Active experiment: none. Completed: primary KAN seed 1729. Pending: matched MLP, affine model, Gaussian parameter estimate.
- Latest checkpoint: `checkpoints/latest.pt`, step 1,000. Best: `checkpoints/best_step_000400.pt`. Preserved pilot: `checkpoints/pilot_step_000031.pt`. Checkpoints are ignored by Git and must remain accessible locally.
- Latest committed milestone: Milestone 1 at `c1da9bd`, receipt `85ef1f2`, both pushed to `origin/main`. Milestone 2 commit and push are pending as this entry is written.

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
- Baselines, finite-sample reference, independent evaluation, additional seeds, and capacity tests remain undone. No final scientific claim has been made.
- The older checkout at `C:\Users\saroy\Codes\New-Denoising-Algorithm` is behind the active project checkout. The active project environment is `C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv` (Python 3.12.14, PyTorch 2.14.1+cpu, CUDA unavailable).

## Next concrete action

Milestone 2 resumed the same optimizer and random states from step 31 and reached step 1,000 without non-finite losses or gradients. Ten periodic 100-step checkpoints, a best checkpoint at step 400, and the final checkpoint were verified. Recomputing fixed validation metrics from both checkpoints reproduced the saved metrics. Best validation squared W2 was `0.00003656` (W2 `0.00605`), mean `0.00416`, and standard deviation `0.19816`; final-step squared W2 was `0.00014622` (W2 `0.01209`). Cumulative training-loop time was `108.90 s` on CPU, excluding process startup and plotting. The best checkpoint plot was inspected: its mapping is decreasing, resembling the decreasing Gaussian quantile map. This is compatible with distribution matching and does not identify an original noise mechanism. See `results/metrics/primary_summary.json` and `results/figures/primary_best_step400.png`.

From this directory in PowerShell, verify the finished run before implementing baselines:

```powershell
& 'C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv\Scripts\python.exe' train.py --status
```

Next implement the parameter-matched MLP, affine generator, and Gaussian parameter-estimation baseline, then evaluate them on the same independent draws as the saved best KAN. Do not retrain the completed KAN without a documented reason.
