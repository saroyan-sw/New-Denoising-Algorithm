import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

from denoise_research.session_guard import PowerStatus


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_session_v3.py"
SPEC = importlib.util.spec_from_file_location("research_session_v3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResearchV3WatchdogTests(unittest.TestCase):
    def test_lock_manifest_rejects_any_changed_dependency(self):
        keys = [
            "config_sha256", "benchmark_sha256", "script_sha256",
            "vendor_provenance_sha256", "model_json_sha256", "deploy_sha256",
            "musiq_checkpoint_sha256", "dists_checkpoint_sha256",
            "vgg_checkpoint_sha256",
        ]
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            record = {}
            for index, key in enumerate(keys):
                path = Path(folder) / f"dependency_{index}"
                path.write_bytes(f"value-{index}".encode())
                paths[key] = path
                record[key] = MODULE.sha(path)
            MODULE.validate_hash_manifest(record, paths)
            paths["script_sha256"].write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "script_sha256"):
                MODULE.validate_hash_manifest(record, paths)

    def test_holdout_binding_rejects_stale_data_and_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "holdout.npz"
            data.write_bytes(b"frozen-data")
            lock_sha = "lock-a"
            manifest = {"lock_sha256": lock_sha, "data_sha256": MODULE.sha(data)}
            complete = dict(manifest)
            MODULE.validate_holdout_binding(manifest, lock_sha, data, complete)
            stale_complete = dict(complete, data_sha256="bad")
            with self.assertRaisesRegex(RuntimeError, "completion"):
                MODULE.validate_holdout_binding(manifest, lock_sha, data, stale_complete)
            data.write_bytes(b"changed-data")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                MODULE.validate_holdout_binding(manifest, lock_sha, data)

    def test_background_poll_and_boundary_writes_are_serialized(self):
        status = PowerStatus(80, False, True, "simulation")
        with tempfile.TemporaryDirectory() as folder:
            watchdog = MODULE.Watchdog(folder, interval=0.001, reader=lambda: status)
            try:
                for index in range(250):
                    watchdog.boundary(f"stress_{index}")
                    time.sleep(0.0005)
            finally:
                watchdog.close()
            self.assertTrue((Path(folder) / "watchdog.json").exists())
            self.assertIsNone(watchdog.reason)


if __name__ == "__main__":
    unittest.main()
