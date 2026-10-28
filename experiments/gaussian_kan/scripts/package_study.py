"""Package the portable study, including local checkpoint artifacts."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / "CHECKPOINT_MANIFEST.json").read_text())
    for entry in manifest["checkpoints"]:
        path = ROOT / entry["path"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Missing or changed checkpoint: {entry['path']}")
    excluded = {".cache", "explorations", "__pycache__", ".ipynb_checkpoints", "outputs"}
    files = [path for path in ROOT.rglob("*") if path.is_file() and
             not any(part in excluded for part in path.relative_to(ROOT).parts)]
    html = output_dir / "Gaussian_KAN_Lab.html"
    if not html.exists():
        raise FileNotFoundError("Execute the notebook before packaging its HTML export")
    files.append(html)
    archive = output_dir / "Gaussian_KAN_Study_Bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            bundle.write(path, "gaussian_kan/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for entry in manifest["checkpoints"]:
            assert hashlib.sha256(bundle.read("gaussian_kan/" + entry["path"])).hexdigest() == entry["sha256"]
        assert bundle.read("gaussian_kan/Gaussian_KAN_Lab.ipynb") == (ROOT / "Gaussian_KAN_Lab.ipynb").read_bytes()
    receipt = {"archive": archive.name, "bytes": archive.stat().st_size,
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "files": len(files), "checkpoint_files": len(manifest["checkpoints"]),
               "verification": "ZIP CRC, every checkpoint SHA-256, and exact notebook bytes passed"}
    (output_dir / "bundle_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
