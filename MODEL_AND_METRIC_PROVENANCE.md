# Model and metric provenance

## Frozen MUSIQ training signal

- Implementation: PyIQA 0.1.16 `musiq`, the KonIQ-10k variant. PyIQA identifies
  it as a no-reference metric with an approximately 0–100 range; higher is better.
  The objective therefore subtracts the standardized score.
- Checkpoint source:
  `https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth`
- Checkpoint SHA-256:
  `e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa`
- Parameters: 27,125,825, all `requires_grad=False`. Training records an isolated
  nonzero `d(MUSIQ)/d(fusion logits)` norm; this proves the frozen scorer influences
  the trainable fusion without updating scorer weights.
- Input: the fused RGB estimate is clipped to `[0,1]` for MUSIQ. In evaluation mode,
  PyIQA maps it to `[-1,1]`, makes aspect-ratio-preserving native, 224-long-side,
  and 384-long-side views, then extracts 32x32 patches at stride 32 with SAME padding.
  For a 128x128 square crop these are 16 + 49 + 144 = 209 image patches, plus the
  learned classification token. The quality head reads the transformer CLS token;
  it does not average per-patch quality predictions.
- Implementation difference: the official TensorFlow preprocessing describes
  Gaussian resize, while PyIQA 0.1.16 uses bicubic interpolation with
  `align_corners=False`. Results here are specifically PyIQA results.
- Domain warning: KonIQ-10k contains authentically distorted, mostly full-resolution
  consumer photographs. Synthetic sigma-25 AWGN on 128px crops is out of domain.
  Direct white-box optimization can exploit no-reference metrics. MC-SURE anchors
  fidelity, lambda is ablated, PSNR selects on development, and DISTS is independent.

Primary references:

- MUSIQ paper and official code: <https://arxiv.org/abs/2108.05997>,
  <https://github.com/google-research/google-research/tree/master/musiq>
- KonIQ-10k: <https://arxiv.org/abs/1910.06180>
- PyIQA: <https://github.com/chaofengc/IQA-PyTorch>

## Independent DISTS evaluation

- Implementation: PyIQA 0.1.16 `dists`, full-reference, lower is better. DISTS is
  called only under `torch.no_grad()` after a candidate is saved. It is never in the
  fusion objective and never selects lambda.
- DISTS checkpoint SHA-256:
  `f5e65c96230b7f6ca995691647d482237e4cab8a50c5c4a5784f219ef0748218`
- Torchvision VGG16 feature checkpoint SHA-256:
  `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`
- DISTS reference: <https://arxiv.org/abs/2004.07728>

The selected-model lock binds both files and the PyIQA version. Summary files report
means, per-seed means, observation standard deviations, source-mean standard
deviations, and paired source-cluster bootstrap intervals for PSNR, SSIM, and DISTS.

## Neural comparator provenance

DnCNN-blind color and DRUNet-color weights come from KAIR release v1.0. Architecture
sources and LICENSE are pinned to KAIR commit
`fc1732f4a4514e42ce15e5b3a1e18c828af47a1e`. Exact URLs, byte counts, and hashes are
verified from `outputs/research_v3/vendor/provenance.json` before inference, lock, and
holdout. The small pinned source files and provenance are copied into the final tracked
artifact package.

## Runtime environment

The exact compact environment record is `environment.lock.txt`. The experiment ran
on CPU under Windows 11; CUDA was unavailable. Peak memory is whole-process Windows
peak working set and is not model-exclusive.
