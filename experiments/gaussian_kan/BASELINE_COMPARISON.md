# Baseline comparison — fixed validation set

All trainable models used independent uniform and target batches, sorted-sample squared W2, Adam at 0.001, batch size 2,048, 1,000 steps, and seed 1729. The selected checkpoint for each neural model minimized squared W2 on the same fixed 8,192-sample validation set. Thus these numbers are model-selection measurements; an independent test set is required for the scientific comparison.

| Generator | Parameters | Validation W2 | Fit/training time, s | Assumption |
| --- | ---: | ---: | ---: | --- |
| KAN `[1,8,8,1]`, best step 400 | 2,960 | 0.00605 | 108.90 | No target-family knowledge |
| MLP `1→52→52→1`, best step 900 | 2,913 | 0.04008 | 3.36 | No target-family knowledge |
| Affine, Adam step 1,000 | 2 | 0.04501 | 2.90 | No target-family knowledge |
| Affine, empirical sorted-quantile fit | 2 | 0.04065 | 0.15 | No target-family knowledge |
| Gaussian parameter fit | 2 estimated moments | 0.00545 | 0.045 | Assumes Gaussian family and uses inverse CDF |
| True inverse-CDF oracle | 0 | 0.00536 | N/A | Uses true target parameters for evaluation only |

The empirical affine fit sorts independent 2,048,000-draw uniform and target training sets and solves the two-parameter least-squares quantile fit. This indicates the affine Adam run had not fully converged by step 1,000. The fitted Gaussian also uses 2,048,000 independent target draws, estimates mean and standard deviation, then applies a Gaussian inverse CDF. These are analytically informed comparators, not distribution-free neural generators.

The MLP's map is nearly linear on the unit interval in the [function and distribution comparison](results/figures/baseline_maps.png). Its result is preserved as observed under the matched protocol; it does not show that all MLP architectures or optimizers underperform. The KAN's best map is decreasing and follows the decreasing quantile reference. [W2 and cost bars](results/figures/baseline_comparison.png) show the accuracy and runtime tradeoff. The full precision values and random seeds are in `results/metrics/baseline_comparison.csv` and `results/metrics/baseline_details.json`.
