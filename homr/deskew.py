"""Global page-level skew correction.

Boxes throughout this project (the OLiMPiC schema, `homr/staff_parsing.py`'s own
`Staff`/`MultiStaff` geometry, `training/omr_datasets/olimpic_repair.py`'s row-by-row
gutter scan) are all plain axis-aligned rectangles. A skewed scan breaks that
assumption for every one of them at once - fixing it once per page, before detection
ever runs, keeps every downstream consumer exactly as simple as it already is, rather
than teaching each one (and the review UI's canvas math) to handle rotated boxes.

The angle itself needs no new detector: `homr.bounding_boxes.AngledBoundingBox` already
normalizes every detected shape's own rotation to a `-45..45` "degrees off horizontal"
convention (used for individual staff-line fragments, clefs, stems - not yet assembled
into a page-wide estimate anywhere). The median across a page's own staff-line
fragments is that estimate; `homr.staff_dewarping` already does a separate, later,
per-staff *curvature* correction (Delaunay-based) right before decoding - this is
upstream of and independent of that, a single rigid rotation for the whole page.
"""

import statistics

import cv2

from homr.main import load_and_preprocess_predictions, predict_symbols
from homr.type_definitions import NDArray

#: Below this many detected staff-line fragments, a page-wide median angle is noise,
#: not signal (a near-blank page, a bad scan) - left uncorrected rather than guessed.
MIN_FRAGMENTS_FOR_SKEW_ESTIMATE = 20

#: Smaller than this isn't worth the resample blur a rotation costs.
SKEW_CORRECTION_THRESHOLD_DEGREES = 0.3


def estimate_skew_angle(image_path: str, segnet_use_gpu: bool = True) -> float | None:
    """Degrees this page's staff lines tilt off horizontal - the median across every
    detected staff-line fragment's own normalized angle, not just one (a single
    fragment can be a false positive; the median across dozens is not). `None` when
    there are too few fragments to trust a page-wide estimate at all.
    """
    predictions, debug = load_and_preprocess_predictions(
        image_path, enable_debug=False, enable_cache=False, segnet_use_gpu=segnet_use_gpu
    )
    symbols = predict_symbols(debug, predictions)
    if len(symbols.staff_fragments) < MIN_FRAGMENTS_FOR_SKEW_ESTIMATE:
        return None
    return statistics.median(fragment.angle for fragment in symbols.staff_fragments)


def rotate_image(image: NDArray, angle_degrees: float) -> NDArray:
    """Rotates the whole image by `angle_degrees` around its own center, expanding the
    canvas so no corner content is cropped off - the border fill is white, matching a
    scanned page's own background rather than leaving black wedges at the corners.
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(height * sin + width * cos)
    new_height = int(height * cos + width * sin)
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def deskew_page_file(image_path: str, segnet_use_gpu: bool = True) -> float:
    """Estimates this page's skew and, if large enough to matter, rewrites the file in
    place corrected. Returns the angle actually corrected (0.0 either because the page
    was already straight, or because there weren't enough staff fragments to estimate
    an angle at all - the caller cannot tell those two apart from the return value
    alone, which is fine: both mean "detection can proceed on the file as it is").
    """
    angle = estimate_skew_angle(image_path, segnet_use_gpu=segnet_use_gpu)
    if angle is None or abs(angle) < SKEW_CORRECTION_THRESHOLD_DEGREES:
        return 0.0
    image = cv2.imread(image_path)
    # No sign flip here despite what that might suggest: `RotatedBoundingBox.angle`'s
    # own convention and `cv2.getRotationMatrix2D`'s (which rotate_image calls
    # directly) turned out to already be each other's inverse, confirmed empirically
    # in test_deskew.py rather than assumed - passing the raw measured angle straight
    # through is what actually straightens a real tilted fragment; negating it here
    # would double the tilt instead of removing it.
    corrected = rotate_image(image, angle)
    cv2.imwrite(image_path, corrected)
    return angle
