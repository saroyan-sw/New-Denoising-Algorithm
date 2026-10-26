# Gaussian noise generator prerequisite

This independent experiment tests whether a cubic B-spline KAN can map uniform latent samples to a Gaussian output law using only unpaired samples and sorted-sample Wasserstein loss. It does not use the Gaussian inverse CDF in training. The repository's denoising experiments are unchanged.

See `EXPERIMENT_PLAN.md` and `EXPERIMENT_STATE.md` for the protocol and current evidence. `RESUME_INSTRUCTIONS.md` is the recovery entry point.

Install `requirements.txt` in a compatible Python environment. The pinned KAN library is [shawcharles/efficient-kan](https://github.com/shawcharles/efficient-kan) at commit `0240ea138fe257d800c9bfa561025e1c44aedc71`, a pure-PyTorch implementation with trainable B-spline edge functions.

From this directory:

```text
python train.py --status
python train.py --max-steps-per-session 200
python train.py --resume latest --max-runtime-seconds 600
```

Python executable and package installation may vary by machine. Checkpoints are local and ignored by Git; lightweight metrics, figures, configuration, and study documentation are version controlled.
