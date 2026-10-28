# Resume instructions

1. Read EXPERIMENT_PLAN.md, EXPERIMENT_STATE.md, and experiment_state.json.
2. Inspect git status --short --branch, git log -5 --oneline, and git remote -v from the repository root.
3. Milestones 0–6 are complete. No experiment needs resuming. Open Gaussian_KAN_Lab.ipynb to inspect the saved results and start optional new approaches.
4. Verify the local artifacts with python scripts/build_checkpoint_manifest.py --verify and the notebook with python tests/test_notebook_artifact.py.
5. For a future interrupted configurable run, use the stored config and original runs root with experiment_runner.py --resume latest. Do not silently restart it or change its configuration.
6. Preserve model binaries separately from Git. New scientific milestones require documentation, verification, commit, and push.

Exact next command from this directory on the current machine:

~~~powershell
& '..\..\.venv\Scripts\python.exe' -m jupyterlab Gaussian_KAN_Lab.ipynb
~~~

For a portable copy, extract outputs/Gaussian_KAN_Study_Bundle.zip. Its top-level gaussian_kan/ folder contains all code, results, notebook, report, and 215 checkpoint files. Install requirements-notebook.txt in a compatible environment and select that interpreter. The HTML export is available under outputs/ for reading without Python.

The local original environment is C:\Users\saroy\.codex\.chatgpt-projects\g-p-6aaad21c30088191ba52fa7797bbe604\New-Denoising-Algorithm\.venv. That path is not hard-coded in the notebook or model code.

The notebook's default execution reads results and runs small checks. Optional new training is disabled until its switch is enabled; it uses separate explorations/ directories, unique run names, explicit resume, and bounded sessions. The completed primary runner refuses to alter a finished checkpoint.
