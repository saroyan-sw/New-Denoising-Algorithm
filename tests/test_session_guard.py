import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from denoise_research.session_guard import atomic_write_json, guarded_checkpoint


class SessionGuardTests(unittest.TestCase):
    def test_atomic_json_retries_transient_windows_access_denial(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            from denoise_research import session_guard
            real_replace = session_guard.os.replace
            attempts = {"count": 0}

            def flaky_replace(source, target):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise PermissionError("simulated file scanner")
                return real_replace(source, target)

            with patch.object(session_guard.os, "replace", side_effect=flaky_replace):
                atomic_write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"ok": True})
            self.assertEqual(attempts["count"], 3)

    def test_simulated_low_battery_saves_before_stopping(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            with self.assertRaisesRegex(RuntimeError, "Battery watchdog stop"):
                guarded_checkpoint(path, {"step": 7}, simulated_percent=14)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["step"], 7)
            self.assertEqual(saved["power"]["battery_percent"], 14)

    def test_checkpoint_can_be_reloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            uninterrupted = 1.0
            for _ in range(4):
                uninterrupted -= 0.1 * (2.0 * uninterrupted)

            interrupted = 1.0
            for step in range(2):
                interrupted -= 0.1 * (2.0 * interrupted)
                guarded_checkpoint(
                    path,
                    {"next_step": step + 1, "weight": interrupted},
                    simulated_percent=80,
                )
            saved = json.loads(path.read_text(encoding="utf-8"))
            resumed = float(saved["weight"])
            for _ in range(int(saved["next_step"]), 4):
                resumed -= 0.1 * (2.0 * resumed)
            self.assertAlmostEqual(resumed, uninterrupted, places=12)


if __name__ == "__main__":
    unittest.main()
