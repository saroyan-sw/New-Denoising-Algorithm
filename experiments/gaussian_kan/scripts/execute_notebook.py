"""Execute every default notebook cell using this environment's actual kernel."""

import hashlib
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    cache = ROOT / ".cache" / "jupyter"
    spec = cache / "kernels" / "gaussian-kan-local"
    spec.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".cache" / "jupyter_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Gaussian KAN validation kernel", "language": "python"
    }))
    os.environ["JUPYTER_PATH"] = str(cache) + (os.pathsep + os.environ["JUPYTER_PATH"] if os.environ.get("JUPYTER_PATH") else "")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(ROOT / ".cache" / "ipython")
    os.environ["JUPYTER_CONFIG_DIR"] = str(ROOT / ".cache" / "jupyter_config")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    path = ROOT / "Gaussian_KAN_Lab.ipynb"
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    notebook = nbformat.read(path, as_version=4)
    begin = time.perf_counter()
    client = NotebookClient(notebook, timeout=180, kernel_name="gaussian-kan-local",
                            resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, ROOT / "outputs" / "notebook_failed.ipynb")
        raise
    nbformat.validate(notebook)
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if errors or any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("Notebook has errors or unexecuted code cells")
    nbformat.write(notebook, path)
    exporter = HTMLExporter(template_name="lab", embed_images=True)
    html, _ = exporter.from_notebook_node(notebook)
    (output_dir / "Gaussian_KAN_Lab.html").write_text(html, encoding="utf-8")
    report = {"notebook": path.name, "status": "passed", "total_cells": len(notebook.cells),
              "executed_code_cells": len(code_cells), "error_outputs": len(errors),
              "execution_seconds": time.perf_counter() - begin,
              "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "python": sys.version.split()[0], "kernel_interpreter": str(Path(sys.executable)),
              "full_study_retrained": False, "optional_new_experiments_enabled": False,
              "recovery_demonstration_executed": True, "artifact_checks_executed": True}
    (ROOT / "results/metrics/notebook_execution.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
