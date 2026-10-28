# Use the Gaussian KAN notebook

1. Open Gaussian_KAN_Lab.ipynb in JupyterLab or a notebook editor and select the project's Python environment.
2. Run the cells from top to bottom. The default path reads completed research, checks its artifacts, and runs a tiny exact recovery demonstration.
3. In section 8, change the saved method list and comparison seed to inspect different models.
4. In section 11, select a recipe, assign a new run name, and edit steps, batch size, seed, or learning rate.
5. Set RUN_NEW_EXPERIMENT = True to train a bounded session. If it stops early, keep the configuration and set RESUME_EXPERIMENT = True before the next session.
6. Add your run name to USER_COMPARISON. Every method in a comparison uses the same fresh latent and target draws.

Completed study runs are under checkpoints/ and runs/; optional notebook runs use ignored explorations/. A changed configuration requires a new run name. Resuming rejects a mismatch. The W1 recipe and alternate MLP activation are available for your experiments but were not part of the completed scientific comparison.

The local project environment already contains the required packages. From the repository root:

~~~powershell
.\.venv\Scripts\python.exe -m jupyterlab experiments\gaussian_kan\Gaussian_KAN_Lab.ipynb
~~~

For a fresh CPU-only Windows environment, first install the tested PyTorch CPU build, then the notebook requirements:

~~~powershell
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-notebook.txt
python -m jupyterlab Gaussian_KAN_Lab.ipynb
~~~

The portable bundle contains the notebook, code, results, and all checkpoint files. A Git-only checkout retains saved notebook outputs, tables, and figures, while live inspection of trained weights needs the bundle. outputs/Gaussian_KAN_Lab.html is a static copy for reading without Python.

To re-execute and validate the notebook using the current interpreter, run python scripts/execute_notebook.py. It uses a local kernel configuration and cache. scripts/build_notebook.py is the authoring source and regenerates the notebook; use it only when deliberately rebuilding its default content.
