"""Validate executed notebook structure, outputs, and execution receipt."""

import hashlib
import json
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "Gaussian_KAN_Lab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    receipt = json.loads((ROOT / "results/metrics/notebook_execution.json").read_text())
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(notebook.cells) == 32 and len(code_cells) == 16
    assert [cell.execution_count for cell in code_cells] == list(range(1, 17))
    assert all(output.output_type != "error" for cell in code_cells for output in cell.outputs)
    pngs = [output for cell in code_cells for output in cell.outputs
            if "data" in output and "image/png" in output.data]
    assert len(pngs) == 10
    assert receipt["status"] == "passed" and receipt["error_outputs"] == 0
    assert receipt["notebook_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert any("RUN_NEW_EXPERIMENT = False" in cell.source for cell in code_cells)
    assert notebook.metadata.kernelspec.name == "python3"
    assert (ROOT / "outputs/Gaussian_KAN_Lab.html").stat().st_size > 100000
    print("32 notebook cells, 16 ordered executions, 10 embedded figures, and matching execution hash verified")


if __name__ == "__main__":
    main()
