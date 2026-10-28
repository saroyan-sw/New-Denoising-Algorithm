# Robustness and capacity — 2026-10-29

The four-run queue was fixed in `config/robustness.json` before execution. Every run used 1,000 Adam steps, batch 2,048, learning rate 0.001, cubic splines, full open-interval uniform inputs, independent Gaussian target samples, and the same fixed validation protocol. Checkpoints were selected by validation alone. All four runs completed without non-finite losses or gradients, with ten periodic checkpoints each.

A fresh 100,000-draw confirmation pair used latent seed `20261029` and target seed `20261030`. The first primary seed was also evaluated on this pair, allowing direct seed comparison. This capacity study is exploratory after the first independent evaluation; no test-based checkpoint reselection was performed.

| Architecture / seed | Parameters | Selected step | Confirmation W2 | Final-step W2 | Interior map RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[1,8,8,1]`, grid 32, seed 1729 | 2,960 | 400 | 0.00624 | 0.00878 | 0.00642 |
| Same, seed 1730 | 2,960 | 975 | 0.00602 | 0.00612 | 0.00600 |
| Same, seed 1731 | 2,960 | 925 | 0.00527 | 0.00594 | 0.00592 |
| `[1,8,1]`, grid 32, seed 1729 | 592 | 700 | 0.00663 | 0.00501 | 0.00660 |
| `[1,8,8,1]`, grid 64, seed 1729 | 5,520 | 450 | 0.00577 | 0.00488 | 0.00572 |

The primary architecture's three-seed confirmation W2 mean was **0.00584**, SD **0.00051**, range **0.00527–0.00624**. All three selected maps were strictly decreasing on the measured interior grid. Ten independent Gaussian pairs had mean W2 **0.00154**, SD **0.00034**; the oracle on this confirmation pair had W2 **0.00203**. These finite samples do not remove the KAN's remaining approximation/optimization error.

The 592-parameter model has one fifth of the primary parameter count and achieved similar central-distribution accuracy. It is the smallest configuration tested, not proof of minimum necessary capacity. Its selected checkpoint still produced no samples beyond absolute value 0.6. The 64-grid model improved seed-1729 0.1%/99.9% quantile errors from approximately `+0.0608/−0.0478` to `+0.0413/−0.0117`; its fraction beyond absolute value 0.6 rose from zero to `0.00157`, still below the theoretical `0.00270`. Increasing resolution helped, but did not solve the tail limitation or reach the sampling reference. Seed 1731 at grid 32 also outperformed seed 1729 at grid 64 in confirmation W2, so optimization variation matters.

Same-session training times were `31.01/31.94 s` for the repeat primary seeds, `18.55 s` for the smaller KAN, and `54.42 s` for grid 64 on CPU with four PyTorch threads. The historical first seed took `108.90 s` in another session/runner; treat timing comparisons across those sessions as indicative rather than controlled benchmarks. The single-seed MLP comparison is not evidence of universal KAN superiority.

Numerical evidence: `results/metrics/robustness_comparison.csv`, `robustness_summary.json`, and each run's immutable config, state, and history under `runs/`. [Confirmation and capacity figure](results/figures/robustness_comparison.png); [orientation-aligned maps and errors](results/figures/capacity_maps.png). Curves are reflected for display only when the learned map decreases; weights and output laws are unchanged.

No wider or deeper KAN was run because the smaller model already handled the central distribution, while the measured remaining issue was tail resolution and noisy optimization. Future work should improve validation averaging, optimization scheduling, and tail diagnostics before a large architecture search.
