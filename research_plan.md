# Bounded session, 2026-10-14 (Asia/Yerevan)

Objective: reusable classical-operator convex fusion trained with frozen MUSIQ and
known-AWGN fidelity; investigate improvement over CBM3D and proximity to pretrained
DnCNN and DRUNet. These are hypotheses, not guaranteed outcomes.

Budget: at most 10 substantial runs, including failed reruns, and four cumulative
active hours. The original segment started at Unix 1791923291 and consumed 2,208.606
seconds before the desktop session was interrupted. The user explicitly resumed on
2026-10-22 at Unix 1792613074.180; the equivalent active-work start is
1792610865.57409, leaving 12,191.394 seconds. Two preliminary runs already count.
CPU only. Stop at battery <=15% on AC or battery. See `architecture_registry.md`
for every pre-registered architecture and its development-only expansion gate.

## Protocol correction before further training

The first pilot used 12 of its 20 development crops for fitting. This conflicts with
the detailed request. Preserve its outputs and mark both runs preliminary/ineligible.
Create a versioned, stratified benchmark of 20 unique archived clean source crops,
selected using only clean-image content statistics with a fixed seed. Use disjoint
source images for reusable-weight training. All are historically exposed project
data, hence development only. Document the departure from full-image DIV2K/CBSD68
evaluation: this checkout lacks full datasets and contains 128x128 archival crops.
The archive contains repeated Kodak sources; deduplicate by source ID before splitting.
Verify exact hashes and conservative perceptual near-duplicate checks.

The five strata are smooth regions, edges, fine texture, repeated structures, and
challenging content. Statistical category proxies are provisional; visually inspect
the clean contact sheet before running models. Selection must not use denoising scores.
Save the frozen selection and three fixed, unclipped sigma=25/255 Gaussian realizations
in benchmark20.json. No training loss receives clean targets. Reference-assisted
oracle fusion uses only development data and is labeled diagnostic.

Final holdout: Urban100 081-100 center 128 crops, a separate structural-domain subset
because CBSD68/Kodak have already informed historical architecture choices. Acquire
and evaluate only after immutable model/config selection. Never claim full Urban100
or standard full-image benchmark performance. Audit source and near-duplicate overlap.

## Comparators and decision criteria

Identity; Gaussian/bilateral/median; NLM with h multipliers 0.7, 0.9, 1.1 selected
on development; official CBM3D at exact sigma; equal fusion; fitted global fusion;
official pretrained color DnCNN (blind) and DRUNet (known sigma). All receive the
same stored float32 noisy arrays, RGB [0,1] reference, no input clipping. Clip outputs
only for displayed/evaluated results. No border shaving. PSNR is primary; RGB SSIM
and frozen DISTS are independent evaluation metrics. DISTS never trains/selects weights.
Report all per-image/seed results, paired source-cluster bootstrap intervals, inference
wall time and process peak RSS (including shared models; not model-exclusive memory).

Retain an improvement claim only if paired holdout PSNR improves over the strongest
classical comparator with a 95% interval excluding zero and no consistent SSIM/DISTS
regression. Neural competitiveness means within 0.2 dB PSNR without perceptual decline;
otherwise report the measured gap. CPU time <=5x strongest classical and peak RSS <=4GB
are exploratory practical targets, not guarantees. Failure does not justify extra runs.

## Hypotheses and gates

H0: first two scalar-SGD pilots were underconverged. Test a projected convex SURE
solution and deterministic recovery control before another learned-IQA run.
H1: original bank lacks diversity (preliminary global oracle gain only ~0.12 dB,
with an optimization/evaluation mismatch). Use a converged, clipped-output validation
oracle; add NLM/CBM3D outputs if their errors show useful diversity. Keep weights global.
H2: MUSIQ regularization trades fidelity for naturalness. Compare lambda=0, .01, .03
under identical train images and normalized SURE, initialized at the fit-only SURE
solution. Record learning curves and inspect every candidate against classical/neural
baselines. The previous 20-step run is also underconverged; do not confuse its outcome
with the representation ceiling.
H3: increase spatial/multiscale/sequence capacity only after a validated oracle and
visible failure support it. Maximum initial sequence depth two; no unrestricted generator.

## Safety and records

Host watchdog polls Windows GetSystemPowerStatus every 30 seconds, independently of
chat, and signals a graceful stop. Periodic atomic checkpoints store fusion parameters,
optimizer and RNG state, exact configuration and next step. Test interrupted recovery
with the actual update routine. Record inapplicable scheduler/scaler explicitly.
All 20 outputs per meaningful candidate must be opened in comparison contact sheets;
inspect representative and worst-delta crops at native scale. Preserve negative results.

Related work: SURE-LET (https://bigwww.epfl.ch/preprints/blu0701p.pdf), MC-SURE
(https://mil.ee.ntu.edu.tw/pdfs/ramani0803.pdf), MUSIQ
(https://openaccess.thecvf.com/content/ICCV2021/html/Ke_MUSIQ_Multi-Scale_Image_Quality_Transformer_ICCV_2021_paper.html),
official implementation https://github.com/google-research/google-research/tree/master/musiq,
PyIQA https://github.com/chaofengc/IQA-PyTorch, BM3D https://webpages.tuni.fi/foi/GCF-BM3D/,
KAIR https://github.com/cszn/KAIR and DPIR https://github.com/cszn/DPIR.
Convex fusion and SURE tuning are established; the plausible contribution is a measured
MUSIQ/fidelity tradeoff and reproducible failure diagnosis, not a novel fusion concept.
