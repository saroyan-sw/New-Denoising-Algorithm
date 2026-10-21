import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_transformer_fusion_session.py"
SPEC = importlib.util.spec_from_file_location("transformer_session", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TransformerSessionProtocolTests(unittest.TestCase):
    def test_simplex_projection_and_identity_reachability(self):
        projected = MODULE._project_simplex(np.array([-1.0, 0.2, 2.0]))
        self.assertTrue(np.all(projected >= 0))
        self.assertAlmostEqual(float(projected.sum()), 1.0, places=7)
        candidates = np.zeros((1, len(MODULE.OPERATORS), 2, 2, 3), dtype=np.float32)
        candidates[:, 0] = 0.25
        weights = np.zeros(len(MODULE.OPERATORS), dtype=np.float32)
        weights[0] = 1.0
        self.assertTrue(np.array_equal(MODULE._fuse(candidates, weights), candidates[:, 0]))

    def test_holdout_gate_rejects_access_before_model_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            MODULE.preflight(Path(folder))
            with self.assertRaisesRegex(RuntimeError, "Holdout gate is closed"):
                MODULE.fetch_holdout(Path(folder))


if __name__ == "__main__":
    unittest.main()
