"""Continue the predeclared bounded robustness queue, skipping completed runs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_runner import ExperimentConfig, run_session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--max-runs-per-session", type=int, default=1)
    parser.add_argument("--max-runtime-seconds", type=float, default=300)
    args = parser.parse_args()
    plan = json.loads((ROOT / "config" / "robustness.json").read_text(encoding="utf-8"))
    count = 0
    matched = False
    for item in plan["runs"]:
        if args.run_id and args.run_id != item["run_id"]:
            continue
        matched = True
        config = ExperimentConfig(**{**plan["shared"], **item})
        run_dir = ROOT / "runs" / config.run_id
        state_path = run_dir / "state.json"
        if state_path.exists() and json.loads(state_path.read_text())["status"] == "completed":
            print(f"Skipping completed run {config.run_id}")
            continue
        resume = (run_dir / "checkpoints" / "latest.pt").is_file()
        result = run_session(config, resume=resume, max_runtime_seconds=args.max_runtime_seconds, track_study=True)
        count += 1
        if result["status"] != "completed" or count >= args.max_runs_per_session:
            break
    if not matched:
        parser.error("run-id is not in the predeclared robustness queue")


if __name__ == "__main__":
    main()
