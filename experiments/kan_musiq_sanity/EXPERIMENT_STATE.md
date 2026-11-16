# KAN/MUSIQ sanity experiment state — 2026-11-16

Current stage: preflight complete. Full hybrid training has NOT started.

User decisions: no clean training references; one noisy observation per scene; iid Gaussian first; one shared unknown noise strength.

Completed:

- Mathematical and empirical controls for independent random subtraction, averaging candidates, zero-amplitude collapse, and a center-excluding noisy-only predictor.
- Sixteen frozen MUSIQ diagnostic scores on four toy patterns, including clean evaluation images, two noisy variants, and flat outputs.
- The metric-ordering gate failed on this toy domain; MUSIQ is excluded from the proposed first training loss.
- Implemented shared and optional conditional spline-KAN prototype branches. Verified masked-target exclusion, finite gradients and updates, and an atomic checkpoint round-trip.

Checkpoint: checkpoints/prototype_shared.pt, one wiring-check update only. The earlier conditional wiring checkpoint is retained locally. These are not trained research results. Checkpoints are ignored by Git.

Evidence: results/mechanical_controls.json, results/mechanical_controls.png, results/musiq_controls.json, results/prototype_shared_checks.json, and the initial conditional results/prototype_checks.json.

Known limitations: no end-to-end denoising result yet; residuals include reconstruction error; independent clean/noise priors need identification constraints; MUSIQ's photographic quality scores did not order the toy scenes by noise fidelity.

Environment reused from the existing project's .venv: Python 3.12, PyTorch 2.14.1+cpu, PyIQA 0.1.16, efficient-kan 0.2.0 pinned by the completed Gaussian study. MUSIQ weights were reused from the local cache, hash e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa. No dependency download was necessary.

Git status: preflight commit/push pending. The previous completed studies are preserved. This work uses a sparse isolated checkout in the chat's outputs folder.

Next action: use ARCHITECTURE.md to build the first masked denoiser baseline under the confirmed constraints. Do not proceed with the literal noisy-input reconstruction loss or assume random prior samples cancel the observed noise. Do not enable MUSIQ training on this toy domain based on its name alone.

Recovery: preflight.py skips completed mathematical/prototype artifacts and resumes MUSIQ scoring from saved rows. It does not automatically run full hybrid training. The cached weight path can be supplied with --musiq-checkpoint; its hash is checked before use.
