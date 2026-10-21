# Resume — development complete, holdout sealed

Read `SESSION_HANDOFF.md`, `EXPERIMENT_REVIEW_GUIDE.md`, and `visual_audit.json`.
All eight substantial training runs are complete; do not train another model. The
registered selection is `v4_lambda_0p01`. A2 spatial, A3 multiscale, and A4 sequential
architectures remain explicitly documented but gated off by the validation-only oracle.

Run the tests, create the immutable model lock, and then execute the one-shot final
holdout exactly as documented in `SESSION_HANDOFF.md`. The holdout must not influence
model choice or architecture. Battery at or below 15% requires saving and stopping.

Current limitations remain: archived 128-pixel crops, historically exposed source
images, category labels based on statistical proxies, and no claim about full-image
Urban100/CBSD68/Kodak performance.
