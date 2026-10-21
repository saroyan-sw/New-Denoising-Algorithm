"""Apply the selected classical-operator fusion to one RGB image."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from research_session_v3 import operator  # noqa: E402

EXPECTED_BANK = ["identity", "bilateral", "nlm_07", "cbm3d"]
DEFAULT_MODEL = ROOT / "outputs" / "research_v3" / "runs" / "v4_lambda_0p01" / "model.json"
DEFAULT_LOCK = ROOT / "outputs" / "research_v3" / "locked_model.json"
RESEARCH_SCRIPT = ROOT / "scripts" / "research_session_v3.py"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input RGB image")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--lock-json", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--allow-unlocked-model",
        action="store_true",
        help="Explicitly bypass lock checks for research only; output is not the selected model",
    )
    args = parser.parse_args()

    model = json.loads(args.model_json.read_text(encoding="utf-8"))
    if not args.allow_unlocked_model:
        lock = json.loads(args.lock_json.read_text(encoding="utf-8"))
        if file_sha256(args.model_json) != lock.get("model_json_sha256"):
            raise RuntimeError("Requested model JSON does not match the selected-model lock")
        if file_sha256(RESEARCH_SCRIPT) != lock.get("script_sha256"):
            raise RuntimeError("Operator implementation does not match the selected-model lock")
        if model.get("run_id") != lock.get("model", {}).get("run_id"):
            raise RuntimeError("Run ID does not match the selected-model lock")
    bank = model.get("bank")
    weights = np.asarray(model.get("weights"), dtype=np.float64)
    if bank != EXPECTED_BANK or weights.shape != (4,):
        raise RuntimeError("Model does not match the locked four-operator architecture")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise RuntimeError("Model weights are not a convex combination")

    with Image.open(args.input) as image:
        noisy = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    start = time.perf_counter()
    estimates = [operator(name, noisy) for name in bank]
    result = np.tensordot(weights.astype(np.float32), np.stack(estimates), axes=(0, 0))
    elapsed = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(result, 0, 1) * 255).astype(np.uint8)).save(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(json.dumps({
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "model_json": str(args.model_json),
        "model_json_sha256": file_sha256(args.model_json),
        "lock_json": None if args.allow_unlocked_model else str(args.lock_json),
        "lock_json_sha256": None if args.allow_unlocked_model else file_sha256(args.lock_json),
        "locked": not args.allow_unlocked_model,
        "run_id": model["run_id"],
        "bank": bank,
        "weights": weights.tolist(),
        "seconds": elapsed,
        "output_sha256": file_sha256(args.output),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} in {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
