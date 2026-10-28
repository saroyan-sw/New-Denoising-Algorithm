# Gaussian generator with a spline KAN

## Objective and scientific boundary

Fit a deterministic generator to transform independent `Z ~ U(0,1)` samples into a distribution close to independent `N ~ Normal(0, 0.2²)` samples. Train using empirical squared one-dimensional Wasserstein-2 loss (sort each sample set independently, then average squared differences). Never pair target and latent draws by index. The target scale is used by the data sampler and evaluator, not by the trainable generator. Distribution matching alone cannot identify the physical noise mechanism, nor does it force an increasing map: a decreasing quantile map has the same output law.

## Milestones and gates

| Milestone | Work | Gate |
| --- | --- | --- |
| 0 | Inspect environment and repository; establish configuration, state, ignore rules, and recovery instructions | `--status` and dependency checks work; commit and push status recorded |
| 1 | Construct `[1,8,8,1]` cubic B-spline KAN; forward/gradient test; 20–50-step pilot; checkpoint and exact resume test | Finite loss and gradients, nonzero update, reload and continuation verified; pilot plots saved |
| 2 | Run 1,000-step primary seed with 2,048 samples per step | Periodic and best checkpoints, validation history, completion verified |
| 3 | Train parameter-matched MLP and affine model; estimate Gaussian mean/std from target sample; evaluate oracle | Baseline metrics and computational costs saved |
| 4 | Evaluate moments, W2, quantiles, monotonicity, interior inverse-CDF fit, tails, and reference sample error | Tables and figures documented with limitations |
| 5 | If justified by evidence and runtime, add two seeds and smaller or finer-grid KAN | Measured stability and capacity/cost tradeoff |
| 6 | Synthesize verified findings and next experiment; create the requested step-by-step Jupyter notebook | Reproducible report, executed notebook with saved results and isolated exploration controls, exact continuation procedure |

Each milestone follows implement, test, run, evaluate, checkpoint, document, commit, push. Stop a run cleanly under its session budget; never mark it complete without verified outputs.

## Planned configurations

Primary: efficient-kan pure-PyTorch B-spline edges, widths `[1,8,8,1]`, 32 grid intervals, order 3, Adam at 0.001, batch 2,048, 1,000 steps, seed 1729. Use CPU in this environment (CUDA unavailable). Fixed validation draws are independent of training draws. Compare MLP at approximately equal trainable parameter count, affine generator, Gaussian parameter estimation, and inverse-CDF oracle. Estimate the sampling floor from two independent Gaussian sets. Additional seeds 1730/1731 and `[1,8,1]` or 64-grid versions are conditional on the primary result.

## Risks and limitations

- The inverse CDF diverges at 0 and 1. Analyze function approximation over `[1e-4, 1-1e-4]`; train on the full open interval, numerically clamped only to avoid exact endpoints.
- Finite batch W2 has sampling noise. Compare with independent target-versus-target samples at the same evaluation size.
- A generated Gaussian law does not prove recovery of the generating process. Function comparisons depend on orientation; report whether the learned map is increasing or decreasing.
- KAN runtime on CPU may be high. Use periodic atomic checkpoints and bounded sessions.
- Large checkpoints remain local and are ignored by Git. A remote-only checkout needs an accessible checkpoint artifact to resume an interrupted run; it can still rerun a completed experiment from code and seed.

## Next action

Milestones 0–6 passed. The final report, executed notebook, checkpoint manifest, and verified portable bundle are complete. No planned run remains unfinished. Open Gaussian_KAN_Lab.ipynb and read FINAL_RESEARCH_REPORT.md. Follow-up experiments need new named configurations, explicit resume, and independent evaluation.

The user-requested notebook explains the setup, checks samples/loss/gradients, demonstrates exact recovery, loads measured results, compares models, and supports isolated configurable experiments. All 16 code cells executed without error, with ten embedded figures and artifact verification.
