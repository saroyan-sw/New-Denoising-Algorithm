from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.pipelines import (
    ComplementaryBandFusion,
    PaperInspiredPipeline,
    complementary_bands,
    overlap_add_apply,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(9)
        self.image = self.rng.random((24, 26, 3), dtype=np.float32)

    def test_complementary_bands_reconstruct_exactly(self):
        low, mid, high = complementary_bands(self.image, 1.0, 2.0)
        np.testing.assert_allclose(low + mid + high, self.image, atol=2e-7)

    def test_unit_band_weights_are_identity(self):
        result = ComplementaryBandFusion().run(self.image)
        np.testing.assert_allclose(result, self.image, atol=2e-7)

    def test_overlap_add_identity_has_no_seams(self):
        result = overlap_add_apply(self.image, lambda patch: patch, patch_size=12, stride=6)
        np.testing.assert_allclose(result, self.image, atol=2e-6)

    def test_paper_pipeline_is_finite_and_bounded(self):
        result = PaperInspiredPipeline().run(self.image)
        self.assertEqual(result.shape, self.image.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()

