# Gaussian noise generator prerequisite

This independent experiment tests whether a cubic B-spline KAN can map uniform latent samples to a Gaussian output law using only unpaired samples and sorted-sample Wasserstein loss. It does not use the Gaussian inverse CDF in training. The repository's denoising experiments are unchanged.

See `EXPERIMENT_PLAN.md` and `EXPERIMENT_STATE.md` for the protocol and current evidence. `RESUME_INSTRUCTIONS.md` is the recovery entry point.

The 31-step pilot passed an exact checkpoint replay test and produced an [initial versus step-30 plot](results/figures/pilot_initial_and_step30.png). At that stage its output was narrower than the target. The [pilot metric record](results/metrics/pilot_validation.json) separates measured values from conclusions.

The 1,000-step primary run is now complete. Its [best-checkpoint plot](results/figures/primary_best_step400.png) shows a decreasing quantile-like map; [verified checkpoint metrics](results/metrics/primary_summary.json) record both best and final validation results.

The [baseline table and interpretation](BASELINE_COMPARISON.md) compare a parameter-matched MLP, trained and empirically fitted affine maps, Gaussian parameter fit, and oracle. [Independent scientific evaluation](SCIENTIFIC_EVALUATION.md) measures the remaining error and tail limitations. [Robustness and capacity results](ROBUSTNESS_AND_CAPACITY.md) cover three primary seeds, a smaller KAN, and finer spline resolution. The final report and step-by-step Jupyter notebook are next.

Install `requirements.txt` in a compatible Python environment. The pinned KAN library is [shawcharles/efficient-kan](https://github.com/shawcharles/efficient-kan) at commit `0240ea138fe257d800c9bfa561025e1c44aedc71`, a pure-PyTorch implementation with trainable B-spline edge functions.

From this directory:

```text
python train.py --status
python train.py --max-steps-per-session 200
python train.py --resume latest --max-runtime-seconds 600
```

Python executable and package installation may vary by machine. Checkpoints are local and ignored by Git; lightweight metrics, figures, configuration, and study documentation are version controlled.
