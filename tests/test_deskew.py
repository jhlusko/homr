import unittest

import cv2
import numpy as np

from homr.bounding_boxes import RotatedBoundingBox
from homr.deskew import rotate_image


def _measure_line_angle(image: np.ndarray) -> float:
    """The same normalized angle convention `homr.deskew.estimate_skew_angle` actually
    relies on (`AngledBoundingBox`'s own -45..45 "degrees off horizontal", inherited by
    `RotatedBoundingBox` - the concrete class real staff-line fragments actually are),
    applied to whatever's the biggest dark contour in `image` - a stand-in for one
    staff-line fragment, not the full segnet pipeline (too heavy for a unit test and
    already exercised elsewhere)."""
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    box = cv2.minAreaRect(biggest)
    return RotatedBoundingBox(box, biggest).angle


def _tilted_line_image(angle_degrees: float, width: int = 400, height: int = 300) -> np.ndarray:
    """A long thin horizontal bar, then tilted by `angle_degrees` - a synthetic stand-in
    for one skewed staff-line fragment, built independently of `rotate_image` itself
    (drawn pre-tilted via its own explicit affine warp) so this test does not just
    check that `rotate_image` inverts itself."""
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, height // 2 - 3), (width - 40, height // 2 + 3), (0, 0, 0), -1)
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=(255, 255, 255))


class TestMeasureLineAngle(unittest.TestCase):
    def test_a_level_line_measures_near_zero(self) -> None:
        image = _tilted_line_image(0.0)
        self.assertLess(abs(_measure_line_angle(image)), 0.5)


class TestRotateImageCorrectsAMeasuredTilt(unittest.TestCase):
    """Confirms the sign `deskew_page_file` actually uses (`rotate_image(image,
    angle)`, the measured angle passed straight through, no negation) really undoes a
    real measured tilt - found empirically, not assumed: `RotatedBoundingBox.angle`'s
    own convention and `cv2.getRotationMatrix2D`'s (used to build these synthetic
    tilts, and what `rotate_image` itself calls) turned out to be exact opposites of
    each other, so the naive "measure X, correct by -X" reasoning is wrong here -
    correcting by the measured angle directly is what actually works.
    """

    def test_a_positive_tilt_is_corrected_by_the_measured_angle_directly(self) -> None:
        tilted = _tilted_line_image(4.0)
        measured = _measure_line_angle(tilted)
        self.assertAlmostEqual(measured, -4.0, delta=1.0)

        corrected = rotate_image(tilted, measured)

        self.assertLess(abs(_measure_line_angle(corrected)), 0.5)

    def test_a_negative_tilt_is_corrected_by_the_measured_angle_directly(self) -> None:
        tilted = _tilted_line_image(-6.0)
        measured = _measure_line_angle(tilted)
        self.assertAlmostEqual(measured, 6.0, delta=1.0)

        corrected = rotate_image(tilted, measured)

        self.assertLess(abs(_measure_line_angle(corrected)), 0.5)

    def test_the_canvas_grows_rather_than_cropping_corners(self) -> None:
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        rotated = rotate_image(image, 30.0)

        self.assertGreater(rotated.shape[0], 100)
        self.assertGreater(rotated.shape[1], 200)


if __name__ == "__main__":
    unittest.main()
