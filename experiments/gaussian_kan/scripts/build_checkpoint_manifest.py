"""Build or verify portable SHA-256 references to all study checkpoints."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    manifest_path = ROOT / "CHECKPOINT_MANIFEST.json"
    if args.verify:
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["checkpoints"]:
            path = ROOT / entry["path"]
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Checkpoint missing or changed: {path}")
        print(f"Verified {len(manifest['checkpoints'])} checkpoint hashes")
        return
    paths = sorted(list((ROOT / "checkpoints").rglob("*.pt")) + list((ROOT / "runs").glob("*/checkpoints/*.pt")))
    entries = []
    for path in paths:
        checkpoint = load_checkpoint(path)
        entries.append({"path": path.relative_to(ROOT).as_posix(), "step": checkpoint["step"],
                        "run_id": checkpoint["config"].get("experiment_id", checkpoint["config"].get("run_id")),
                        "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path.write_text(json.dumps({"format": 1, "checkpoints": entries}, indent=2) + "\n")
    packages = ["torch", "numpy", "scipy", "matplotlib", "pandas", "PyYAML", "efficient-kan", "nbformat", "nbclient", "ipykernel", "jupyterlab"]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in packages},
                   "kan_source": "https://github.com/shawcharles/efficient-kan/tree/0240ea138fe257d800c9bfa561025e1c44aedc71"}
    (ROOT / "environment_study.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(f"Manifest written for {len(entries)} checkpoints ({sum(entry['bytes'] for entry in entries):,} bytes)")


if __name__ == "__main__":
    main()
