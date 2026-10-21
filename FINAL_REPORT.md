# Final report: frozen-IQA classical-operator fusion

## Outcome

The selected deployable model is `v4_lambda_0p01`, a four-weight global convex
fusion of identity, bilateral, NLM (`h=0.7 sigma`), and color BM3D. Its weights are:

```text
identity  0.0000550514
bilateral 0.0019913730
NLM-0.7   0.0035475527
CBM3D     0.9944059849
```

The evidence does **not** support a meaningful improvement over the strongest
classical method. On the sealed holdout, the fusion gains 0.00293 dB PSNR over
CBM3D (paired source-cluster bootstrap 95% CI 0.00178 to 0.00414) and improves
DISTS by 0.000099 (CI 0.000058 to 0.000142), but it consistently reduces SSIM by
0.0000483 (CI -0.0000809 to -0.0000209). The preregistered claim required no
consistent SSIM or DISTS regression, so it fails despite the positive PSNR interval.
The result is also too small to be practically visible.

The model is not consistently competitive with the named neural baselines. It trails
DRUNet by 1.029 dB on development and 1.499 dB on holdout. It trails DnCNN by
0.412 dB on development, although it beats blind DnCNN by 1.037 dB on the structural
Urban100 holdout. That reversal is a domain-dependent result, not general superiority.

## Protocol

- Noise: deterministic unclipped RGB AWGN, sigma 25/255; three fixed draws per image.
- Development: 20 fixed 128x128 archived crops, stratified using clean-content proxies
  and source-disjoint from 16 fitting crops. Exact paths and hashes are frozen in
  `benchmark20.json`.
- Holdout: Urban100 images 081-100, center 128x128 crops, acquired only after the
  selected model/configuration/script/checkpoints were hash-locked. It is a custom
  subset and does not represent standard full-image Urban100 performance.
- Objective: MC-SURE divided by noise variance minus lambda times standardized frozen
  MUSIQ/KonIQ score. MUSIQ's 27,125,825 parameters were frozen; an isolated nonzero
  gradient from MUSIQ to the four logits was verified.
- Selection: highest development PSNR among the two v4 positive-lambda candidates.
  PSNR was primary; SSIM and frozen DISTS were evaluation-only.
- Statistics: means over 20 sources x three paired noise draws. Confidence intervals
  resample the 20 source-level means 10,000 times, retaining the paired noise draws.

## Fair comparison results

All methods received the exact same stored float32 noisy arrays, operator-native
boundaries, no input clipping, output clipping only for scoring/display, and no border
shaving. Runtime is mean CPU seconds per 128x128 image; fusion runtime includes every
member operator. DISTS is lower-is-better.

### Fixed development benchmark

| Method | PSNR dB | SSIM | DISTS | s/image |
|---|---:|---:|---:|---:|
| Identity | 20.6773 | 0.428516 | 0.285611 | 0.0009 |
| Gaussian 0.7 | 25.7387 | 0.652029 | 0.239271 | 0.0036 |
| Gaussian 1.1 | 25.7543 | 0.682375 | 0.262323 | 0.0050 |
| Bilateral | 27.7075 | 0.739777 | 0.225887 | 0.0322 |
| Median 3x3 | 24.5469 | 0.588302 | 0.261155 | 0.0089 |
| NLM 0.7 | 29.0572 | 0.809947 | 0.210342 | 0.0235 |
| NLM 0.9 | 28.3590 | 0.778513 | 0.228700 | 0.0294 |
| NLM 1.1 | 27.5949 | 0.746443 | 0.244197 | 0.0282 |
| CBM3D | 31.5454 | 0.893008 | 0.160786 | 3.9537 |
| Equal fusion | 27.5918 | 0.709827 | 0.199497 | 4.0103 |
| **Selected fusion** | **31.5478** | **0.892963** | **0.160590** | **4.0104** |
| DnCNN-blind | 31.9602 | 0.901101 | 0.149958 | 0.3670 |
| DRUNet sigma-known | 32.5766 | 0.910391 | 0.142797 | 1.3255 |

### Sealed Urban100 081-100 holdout

| Method | PSNR dB | SSIM | DISTS | s/image |
|---|---:|---:|---:|---:|
| Identity | 20.6257 | 0.564817 | 0.266409 | 0.0004 |
| Gaussian 0.7 | 24.3019 | 0.718136 | 0.235635 | 0.0044 |
| Gaussian 1.1 | 22.6659 | 0.671490 | 0.266861 | 0.0071 |
| Bilateral | 26.4634 | 0.796241 | 0.215239 | 0.0396 |
| Median 3x3 | 22.7299 | 0.629923 | 0.262835 | 0.0096 |
| NLM 0.7 | 27.5422 | 0.832955 | 0.170856 | 0.0232 |
| NLM 0.9 | 26.6281 | 0.794459 | 0.196644 | 0.0243 |
| NLM 1.1 | 25.4989 | 0.754155 | 0.219277 | 0.0245 |
| CBM3D | 30.5806 | 0.922107 | 0.121366 | 2.7462 |
| Equal fusion | 27.0944 | 0.794138 | 0.176854 | 2.8094 |
| **Selected fusion** | **30.5835** | **0.922059** | **0.121267** | **2.8095** |
| DnCNN-blind | 29.5462 | 0.903945 | 0.131995 | 0.1151 |
| DRUNet sigma-known | 32.0830 | 0.940398 | 0.092790 | 0.3836 |

Complete per-seed means, sample standard deviations, source-mean standard deviations,
per-image rows, and paired intervals are stored in the tracked final artifacts. Peak
working set was about 1.47 GB during development and 1.58 GB during holdout; this is
whole-process peak memory, not model-exclusive memory.

## Complementarity and architecture decision

The initial bank's global reference oracle exceeded CBM3D by only 0.0068 dB. Replacing
NLM-0.9 with development-selected NLM-0.7 increased this to 0.0140 dB, but the
per-image oracle added only 0.0289 dB over the global weights and only one of 20 images
had at least 0.1 dB headroom. The registered A2 spatial gate required 0.10 dB mean
headroom and five visibly complementary images. It failed, so spatial, multiscale,
and short sequential models were not run. This avoided spending capacity on operator
outputs whose errors were overwhelmingly correlated.

Eight substantial runs counted against the limit: two preserved but ineligible pilots,
three original-bank runs, and three NLM-corrected runs. The SURE-only corrected control
was the best fusion on development at 31.55998 dB, but it had a larger consistent SSIM
regression and was not eligible for the positive-IQA selection rule. All 20 images and
the worst panel were opened after every meaningful candidate. The final holdout's
selected, DnCNN, and equal-fusion pages and worst panels were also opened. Findings are
itemized in `experiments.csv` and `visual_audit.json`.

## Failures and lessons

- Two early scalar-SGD pilots were underconverged; one stayed near identity and one
  oversmoothed. They also reused development sources and are explicitly ineligible.
- A concurrent watchdog/file-replace race interrupted MC-SURE computation. Serialized
  heartbeat writes, retrying atomic replacement, and exact cache resume fixed it.
- Uniform fusion leaves obvious colored residual noise. Learned solutions converge
  almost completely to CBM3D because the operator bank has little useful diversity.
- Larger MUSIQ weight increasingly collapses to CBM3D; no visual texture recovery was
  observed. This is consistent with KonIQ-to-synthetic-AWGN domain mismatch and warns
  against trusting a no-reference score without fidelity and reference evaluation.
- DnCNN's development-to-holdout reversal demonstrates that one benchmark cannot
  establish broad neural competitiveness.

## Reproduce and deploy

With CPython 3.12, create the recorded pinned core environment, verify the hashes in
the locked artifacts, and run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-research-lock.txt
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts\apply_locked_fusion.py input.png output.png
```

The command rejects a model or operator implementation that differs from the lock and
writes the denoised PNG plus a JSON sidecar with lock/input/model/output hashes,
weights, and elapsed time. The tracked rows, summaries, views, source manifest, and
lock make a fresh checkout auditable, but large noisy arrays, source images, and
pretrained checkpoints are intentionally omitted. Therefore a fresh checkout cannot
recompute the completed holdout without reacquiring those inputs; it can deploy the
selected classical model after installing the pinned dependencies. An existing
experiment checkout with its ignored caches can revalidate via `SESSION_HANDOFF.md`.
Do not reopen model selection from the final holdout.

## Literature and provenance

The design follows established [SURE-LET](https://bigwww.epfl.ch/preprints/blu0701p.pdf)
and [Monte-Carlo SURE](https://mil.ee.ntu.edu.tw/pdfs/ramani0803.pdf) ideas rather
than claiming convex fusion as novel. The frozen transformer prior is
[MUSIQ](https://arxiv.org/abs/2108.05997), trained on the
[KonIQ-10k](https://arxiv.org/abs/1910.06180) authentic-distortion dataset through
[PyIQA](https://github.com/chaofengc/IQA-PyTorch). Neural comparators use pinned
[KAIR](https://github.com/cszn/KAIR) code and official release weights. Independent
evaluation uses [DISTS](https://arxiv.org/abs/2004.07728). Exact URLs, versions,
hashes, preprocessing details, and domain warnings are in
`MODEL_AND_METRIC_PROVENANCE.md` and the final lock.
