import unittest

import numpy as np

from training.omr_datasets.contrast_normalize import contrast, ink_fraction, normalize


def _faint_page(height: int = 100, width: int = 200) -> np.ndarray:
    """Uniform light gray with a few faint dark strokes - a washed-out scan."""
    page = np.full((height, width), 200, dtype=np.uint8)
    page[40:42, 20:180] = 150
    return page


def _crisp_page(height: int = 100, width: int = 200) -> np.ndarray:
    page = np.full((height, width), 255, dtype=np.uint8)
    page[40:42, 20:180] = 0
    return page


class TestInkFraction(unittest.TestCase):
    def test_a_blank_page_has_no_ink(self) -> None:
        self.assertEqual(ink_fraction(np.full((10, 10), 255, dtype=np.uint8)), 0.0)

    def test_a_black_page_is_all_ink(self) -> None:
        self.assertEqual(ink_fraction(np.zeros((10, 10), dtype=np.uint8)), 1.0)


class TestContrast(unittest.TestCase):
    def test_a_flat_page_has_no_contrast(self) -> None:
        self.assertEqual(contrast(np.full((10, 10), 128, dtype=np.uint8)), 0.0)

    def test_more_extreme_values_give_more_contrast(self) -> None:
        mild = np.full((20, 20), 128, dtype=np.uint8)
        mild[:10] = 150
        sharp = np.full((20, 20), 128, dtype=np.uint8)
        sharp[:10] = 250

        self.assertLess(contrast(mild), contrast(sharp))


class TestNormalize(unittest.TestCase):
    """The transform this file exists to evaluate: does it help the faint page without
    hurting the crisp one?"""

    def test_a_faint_page_gains_contrast(self) -> None:
        faint = _faint_page()

        self.assertGreater(contrast(normalize(faint)), contrast(faint))

    def test_a_crisp_page_is_not_pushed_into_noise(self) -> None:
        # The risk this module exists to catch: an aggressive transform that fixes faint
        # scans by damaging clean ones. Contrast should not drop on an already-sharp page.
        crisp = _crisp_page()

        self.assertGreaterEqual(contrast(normalize(crisp)), contrast(crisp) - 5)

    def test_output_is_still_a_valid_image(self) -> None:
        result = normalize(_faint_page())

        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue((result >= 0).all() and (result <= 255).all())

    def test_the_clip_limit_bounds_how_far_a_flat_region_is_stretched(self) -> None:
        # A tile with almost no variation (background paper) should not be amplified into
        # visible noise merely because CLAHE operates locally.
        flat = np.full((64, 64), 250, dtype=np.uint8)
        flat[30, 30] = 245  # one pixel of near-invisible texture

        result = normalize(flat, clip_limit=2.0)

        self.assertLess(int(result.max()) - int(result.min()), 40)


if __name__ == "__main__":
    unittest.main()
