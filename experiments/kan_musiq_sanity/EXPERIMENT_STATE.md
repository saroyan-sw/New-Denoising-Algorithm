# KAN/MUSIQ sanity experiment state — 2026-11-16

Current stage: toy preflight and fixed natural-photo metric calibration complete. Full hybrid training has NOT started.

User decisions: no clean training references; one noisy observation per scene; iid Gaussian first; one shared unknown noise strength.

Completed:

- Mathematical and empirical controls for independent random subtraction, averaging candidates, zero-amplitude collapse, and a center-excluding noisy-only predictor.
- Sixteen frozen MUSIQ diagnostic scores on four toy patterns, including clean evaluation images, two noisy variants, and flat outputs.
- The metric-ordering gate failed on this toy domain; MUSIQ is excluded from the proposed first training loss.
- Implemented shared and optional conditional spline-KAN prototype branches. Verified masked-target exclusion, finite gradients and updates, and an atomic checkpoint round-trip.
- Created and inspected an image explaining the completed toy checks. It is an illustration; none of its thumbnails is a saved trained-model reconstruction.
- Evaluated a predeclared four-photo natural-image panel at original sizes, using three added-noise seeds and 72 saved scores. All four mean scores decreased monotonically with added noise. The predeclared natural-domain auxiliary-loss gate passed; no image was selected based on score.
- Extended the prototype to RGB. Verified its 9,271-parameter forward pass, target masking, gradients, one update per branch, and checkpoint save/load. The shared KAN remains 2,960 parameters.
- Verified frozen MUSIQ has a finite nonzero natural-image input gradient; no metric optimization was performed.

Checkpoint: checkpoints/prototype_shared.pt, one wiring-check update only. The earlier conditional wiring checkpoint is retained locally. These are not trained research results. Checkpoints are ignored by Git.

Evidence: results/mechanical_controls.json, results/mechanical_controls.png, results/musiq_controls.json, results/prototype_shared_checks.json, and the initial conditional results/prototype_checks.json.

Known limitations: no end-to-end denoising result yet; residuals include reconstruction error; independent clean/noise priors need identification constraints; MUSIQ's photographic quality scores did not order toy scenes by noise fidelity. The positive natural-photo calibration is limited to four fixed images and does not justify standalone score optimization. Smoothing reduced reference MSE but also reduced MUSIQ, showing a fidelity/detail trade-off.

Environment reused from the existing project's .venv: Python 3.12, PyTorch 2.14.1+cpu, PyIQA 0.1.16, efficient-kan 0.2.0 pinned by the completed Gaussian study. MUSIQ weights were reused from the local cache, hash e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa. No dependency download was necessary.

Git status: preflight commit 4165522 pushed successfully to origin/main (e3e2cdb..4165522). The previous completed studies are preserved. This work uses a sparse isolated checkout in the chat's outputs folder. Full hybrid training remains not started.

Next action: follow NATURAL_IMAGES_AND_REVISED_ARCHITECTURE.md to build the masked RGB baseline, then identify/anchor the shared noise prior and test a small frozen-MUSIQ auxiliary-loss ablation. The literal noisy-input reconstruction loss remains invalid. Posterior coupling remains proposed and has not been implemented.

Recovery: preflight.py skips completed mathematical/prototype artifacts and resumes MUSIQ scoring from saved rows. It does not automatically run full hybrid training. The cached weight path can be supplied with --musiq-checkpoint; its hash is checked before use.
