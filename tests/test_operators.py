from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from denoise_research.metrics import mse, psnr, ssim
from denoise_research.operators import bilateral_filter, gaussian_blur, gaussian_kernel, median_filter


class OperatorTests(unittest.TestCase):
    def test_gaussian_kernel_is_normalized_and_symmetric(self):
        kernel = gaussian_kernel(7, 1.5)
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1], atol=1e-7)

    def test_gaussian_blur_preserves_constant_image(self):
        image = np.full((19, 23, 3), 0.37, dtype=np.float32)
        np.testing.assert_allclose(gaussian_blur(image, 1.2), image, atol=1e-6)

    def test_bilateral_preserves_constant_image(self):
        image = np.full((11, 13, 3), 0.42, dtype=np.float32)
        result = bilateral_filter(image, radius=2, sigma_spatial=1.5, sigma_range=0.1)
        np.testing.assert_allclose(result, image, atol=1e-6)

    def test_median_filter_removes_isolated_impulse(self):
        image = np.zeros((9, 9, 3), dtype=np.float32)
        image[4, 4] = 1.0
        result = median_filter(image, size=3)
        np.testing.assert_array_equal(result, np.zeros_like(image))

    def test_identity_metrics(self):
        rng = np.random.default_rng(4)
        image = rng.random((32, 32, 3), dtype=np.float32)
        self.assertEqual(mse(image, image), 0.0)
        self.assertEqual(psnr(image, image), float("inf"))
        self.assertAlmostEqual(ssim(image, image), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
