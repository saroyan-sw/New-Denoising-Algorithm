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
