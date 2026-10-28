# Independent scientific evaluation

## Protocol

Model selection finished before this evaluation: KAN best at step 400, MLP best at step 900, affine Adam at step 1,000. A separate 100,000-draw uniform set and independent 100,000-draw Gaussian target set were generated using seeds `824501` and `912877`. All methods use the same test draws. Ten additional independent Gaussian-versus-Gaussian pairs of size 100,000 estimate finite-sample W2 variation. Function comparisons use a 10,000-point grid on `[1e-4, 1-1e-4]`; these oracle calculations are evaluation only.

## Distributional results

| Generator | Test W2 | Mean error | Std error | Parameters | Training-loop time |
| --- | ---: | ---: | ---: | ---: | ---: |
| KAN best, step 400 | 0.00776 | +0.00431 | −0.00067 | 2,960 | 108.90 s |
| KAN final, step 1,000 | 0.00748 | −0.00631 | −0.00144 | 2,960 | same run |
| MLP best, step 900 | 0.04294 | +0.00322 | −0.00650 | 2,913 | 3.36 s |
| Affine Adam, step 1,000 | 0.04765 | −0.00075 | −0.02457 | 2 | 2.90 s |
| Affine empirical sorted-quantile fit | 0.04300 | −0.00047 | −0.00435 | 2 | 0.15 s fit |
| Gaussian parameter fit | 0.00165 | −0.00054 | +0.00005 | 2 estimated moments | 0.045 s fit |
| True inverse-CDF oracle | 0.00166 | −0.00042 | +0.00003 | 0 | evaluation only |

The independent Gaussian-versus-Gaussian W2 reference averaged **0.00155** (standard deviation **0.00034**) over ten pairs. Thus the KAN substantially improved on this MLP and affine protocol, yet did not reach the finite-sample reference. The Gaussian parameter fit and true oracle are near that reference, as expected from their Gaussian knowledge. All values are in `results/metrics/independent_test_metrics.csv` and `sampling_floor.json`.

The validation-selected KAN checkpoint was slightly worse than the final checkpoint on this one independent test set. We report both; the test set was not used to replace the preselected checkpoint. This fluctuation suggests a more robust validation protocol for future runs.

## Learned transformation and tails

The step-400 KAN map was strictly **decreasing** on the interior evaluation grid (9,999 of 9,999 adjacent differences were negative). Against the decreasing reference `-0.2 Φ⁻¹(z)`, its interior RMSE was **0.00642**; against the increasing reference it was **0.39905**. The final map was also strictly decreasing, with interior RMSE **0.00786** against the decreasing reference. Distribution matching can learn either orientation. It has not identified an underlying stochastic noise mechanism.

The inverse CDF diverges at the endpoints; these finite models flatten. At the 0.1% and 99.9% output quantiles, the step-400 KAN errors against the theoretical Gaussian were **+0.06055** and **−0.04530**. Among 100,000 generated values it produced none with absolute value above 0.6, versus theoretical probability **0.00270** (about 270 expected). The final KAN produced a fraction **0.00193** beyond 0.6, still below the target. Tails remain the clearest miss despite close central histograms. See `results/metrics/quantile_errors.csv`, `tail_fractions.csv`, and `function_diagnostics.json`.

## Figures

- [Independent target and generator histograms](results/figures/independent_histograms.png)
- [Training and validation curves](results/figures/validation_curves.png)
- [Function versus both inverse-CDF orientations](results/figures/inverse_cdf_comparison.png)
- [Independent quantile–quantile plot](results/figures/independent_qq.png)
- [Tail quantile errors](results/figures/tail_quantile_errors.png)
- [Initialization and intermediate distributions](results/figures/distribution_snapshots.png)

## Computational interpretation

The KAN used about **32×** the MLP training-loop time (`108.90` versus `3.36` seconds) and took about `0.75` seconds for one warmed 100,000-sample CPU inference versus `0.011` seconds for the MLP. The inference timing is one measurement, so treat its ratio as indicative. On a known Gaussian target, estimating two parameters is both more accurate and far cheaper. The KAN result demonstrates useful distribution-free spline expressivity for this prerequisite, but does not justify KAN as the preferred way to generate known Gaussian noise.

At this milestone, only one KAN and one MLP seed had been evaluated. The subsequent [robustness and capacity study](ROBUSTNESS_AND_CAPACITY.md) adds two KAN seeds, a smaller KAN, and a finer-grid KAN. The comparison remains conditional on the chosen architectures, initialization, Adam settings, and sample budget. No corrupted-image process has been inferred or tested here.
