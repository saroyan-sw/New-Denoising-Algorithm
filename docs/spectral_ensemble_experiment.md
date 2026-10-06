# Spectral filter-ensemble experiment

## Question

Can no-reference naturalness scores select a useful **global convex mixture** of deterministic denoisers without using clean targets to choose that mixture?

This is a controlled test of the proposed book-reader analogy. It is not yet a claim that the learned numbers are mathematically optimal denoising coefficients.

## Matrix and alternating update

For calibration image (i), candidate filter/configuration (j), and no-reference metric (k), compute the raw score

\[
s^{(k)}_{ij}=Q_k(F_j(y_i)).
\]

Raw NIQE, BRISQUE, and other metric scales must not be added directly. For each image and metric, the implementation converts candidates to within-row positive affinities (A^{(k)}_{ij}), with rank conversion as the default. Metric affinities are averaged:

\[
A=\frac{1}{K}\sum_k A^{(k)}.
\]

Starting from uniform image and filter weights, the damped mutual-reinforcement update is

\[
v \leftarrow \operatorname{norm}(A^T u), \qquad
u \leftarrow \operatorname{norm}(Av),
\]

followed by a small uniform teleportation term. The fused restoration is the convex combination

\[
\hat x_i=\sum_j v_j F_j(y_i), \qquad v_j\geq0,\quad \sum_jv_j=1.
\]

The fitted vector (v) is global: it is learned on DIV2K validation and then frozen for CBSD68 and Kodak24. Recomputing it from benchmark test scores is a different, transductive experiment and must be labeled as such.

## Initial protocol

- Signal: clean RGB image scaled to [0,1].
- Degradation: additive white Gaussian noise, initially sigma 25/255, without clipping the noisy observation.
- Calibration: one seeded 256x256 crop from each of 100 DIV2K validation images.
- Candidate bank: identity; Gaussian sigma 0.6, 1.0, 1.5; median 3 and 5; mild, medium, and strong bilateral configurations.
- Selection signals: BRISQUE and NIQE by default, converted separately to within-image ranks. Both are lower-is-better.
- Locked tests: full-resolution CBSD68 and Kodak24.
- Report: per-image and mean PSNR, SSIM, MSE, and MAE for every candidate, uniform fusion, and spectral fusion; save raw NR scores, affinities, learned coefficients, seed, crop rule, and convergence status.

Identity is a necessary control. Multiple strengths from one family are distinct columns because the coefficient experiment is selecting operators and hyperparameters together.

## Interpretation and validity checks

A successful result requires more than improving BRISQUE/NIQE on the calibration images. The frozen fusion should improve held-out PSNR/SSIM over simple Gaussian/bilateral filtering and should remain stable on both test sets. Always report the oracle convex or best-candidate result separately if it is later computed using clean targets; it is an upper-bound diagnostic, not a valid unsupervised method.

Major risks are metric exploitation, correlated metrics, a redundant candidate bank, and content-dependent best filters being compressed into one global vector. Compare the learned vector against uniform averaging, each individual filter, and clean-target least-squares/oracle weights. If NR-guided weights fail but oracle weights work, the quality signal is the problem. If neither works, linear global fusion or the filter bank is the likely bottleneck.

## Commands

Install optional metrics first:

```powershell
pip install -e ".[torch]"
```

Fit once on calibration data:

```powershell
python scripts/run_spectral_ensemble.py fit data/DIV2K_valid --output-dir outputs/spectral_sigma25 --sigma 25 --metric brisque:lower --metric niqe:lower
```

Apply the same saved vector to each held-out set:

```powershell
python scripts/run_spectral_ensemble.py evaluate data/CBSD68 --weights outputs/spectral_sigma25/weights.json --output-dir outputs/cbsd68_sigma25
python scripts/run_spectral_ensemble.py evaluate data/Kodak24 --weights outputs/spectral_sigma25/weights.json --output-dir outputs/kodak24_sigma25
```
