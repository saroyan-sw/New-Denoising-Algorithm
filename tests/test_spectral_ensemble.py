from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.experiment_utils import add_awgn, random_crop
from denoise_research.filter_bank import apply_filter_bank, default_filter_bank
from denoise_research.nr_metrics import parse_metric_spec
from denoise_research.spectral_ensemble import (
    combine_affinities,
    fit_spectral_weights,
    fuse_candidates,
    metric_to_affinity,
)


class SpectralEnsembleTests(unittest.TestCase):
    def test_rank_affinity_respects_metric_direction(self):
        scores = np.array([[3.0, 1.0, 2.0], [6.0, 2.0, 4.0]])
        lower = metric_to_affinity(scores, lower_is_better=True)
        higher = metric_to_affinity(scores, lower_is_better=False)
        np.testing.assert_array_equal(np.argmax(lower, axis=1), np.array([1, 1]))
        np.testing.assert_array_equal(np.argmax(higher, axis=1), np.array([0, 0]))

    def test_spectral_solver_favors_consistently_best_column(self):
        affinity = np.array(
            [[0.2, 1.0, 0.5], [0.1, 1.0, 0.4], [0.3, 1.0, 0.6]], dtype=np.float64
        )
        result = fit_spectral_weights(affinity)
        self.assertTrue(result.converged)
        self.assertEqual(int(np.argmax(result.filter_weights)), 1)
        self.assertAlmostEqual(float(result.filter_weights.sum()), 1.0)
        self.assertAlmostEqual(float(result.image_weights.sum()), 1.0)

    def test_affinity_combination_and_fusion(self):
        first = np.array([[1.0, 0.5]])
        second = np.array([[0.5, 1.0]])
        np.testing.assert_allclose(combine_affinities([first, second]), [[0.75, 0.75]])
        candidates = {
            "black": np.zeros((3, 3, 3), dtype=np.float32),
            "white": np.ones((3, 3, 3), dtype=np.float32),
        }
        fused = fuse_candidates(candidates, {"black": 1.0, "white": 3.0})
        np.testing.assert_allclose(fused, 0.75)

    def test_default_filter_bank_has_unique_shape_preserving_outputs(self):
        rng = np.random.default_rng(7)
        image = rng.random((12, 13, 3), dtype=np.float32)
        bank = default_filter_bank()
        outputs = apply_filter_bank(image, bank)
        self.assertEqual(len(outputs), len(bank))
        self.assertEqual(len(outputs), len(set(outputs)))
        for output in outputs.values():
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(np.isfinite(output).all())

    def test_seeded_crop_and_noise_are_reproducible(self):
        image = np.arange(20 * 21 * 3, dtype=np.float32).reshape(20, 21, 3) / 1260.0
        rng_a = np.random.default_rng(11)
        rng_b = np.random.default_rng(11)
        crop_a = random_crop(image, 12, rng_a)
        crop_b = random_crop(image, 12, rng_b)
        np.testing.assert_array_equal(crop_a, crop_b)
        np.testing.assert_array_equal(add_awgn(crop_a, 25, rng_a), add_awgn(crop_b, 25, rng_b))

    def test_metric_spec_requires_explicit_direction(self):
        spec = parse_metric_spec("brisque:lower")
        self.assertEqual(spec.name, "brisque")
        self.assertTrue(spec.lower_is_better)
        with self.assertRaises(ValueError):
            parse_metric_spec("niqe")


if __name__ == "__main__":
    unittest.main()
