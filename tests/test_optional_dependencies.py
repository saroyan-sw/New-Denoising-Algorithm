from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research import nr_metrics, torch_optional


class OptionalDependencyTests(unittest.TestCase):
    def test_torch_module_is_importable_without_torch(self):
        if torch_optional.torch is None:
            with self.assertRaisesRegex(RuntimeError, "requires PyTorch"):
                torch_optional.require_torch()
        else:
            torch_optional.require_torch()

    def test_no_reference_metrics_fail_with_an_actionable_message(self):
        try:
            nr_metrics._dependencies()
        except nr_metrics.OptionalMetricDependencyError as error:
            self.assertIn("pip install -e .[torch]", str(error))


if __name__ == "__main__":
    unittest.main()

