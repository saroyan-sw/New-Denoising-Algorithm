# Study artifacts

- Gaussian_KAN_Lab.ipynb: executed interactive notebook, 32 cells, 16 code cells, 10 embedded figures.
- FINAL_RESEARCH_REPORT.md: conclusions, measured comparisons, limitations, and next experiment.
- NOTEBOOK_GUIDE.md: launch instructions and new-experiment controls.
- CHECKPOINT_MANIFEST.json: hashes, sizes, steps, and run IDs for 215 local checkpoint files.
- environment_study.json and requirements-study-lock.txt: verified scientific environment.
- results/metrics/: full precision tables, function/tail diagnostics, sampling references, and notebook execution receipt.
- results/figures/: scientific plots used in the report.
- runs/: immutable configurations, per-run state, history, and summaries for robustness experiments.
- checkpoints/ and runs/*/checkpoints/: local model, optimizer, and random-state artifacts, ignored by Git.
- outputs/Gaussian_KAN_Lab.html: static notebook reading copy.
- outputs/Gaussian_KAN_Study_Bundle.zip: portable code, notebook, report, results, and all study checkpoints.
- outputs/bundle_manifest.json: archive size/hash and verification receipt.

The ZIP and model binaries remain local. The repository tracks their lightweight manifests, notebook outputs, configuration, metrics, figures, and documentation. Recreate the bundle with python scripts/package_study.py when the checkpoint files are present. The package script checks every checkpoint hash and verifies ZIP contents.

The notebook's optional experiments use explorations/, which is isolated from the completed study and ignored by Git. They are not added to the published evidence automatically.
