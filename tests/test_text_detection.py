"""`homr.text_detection` is a reimplementation, so its tests are mostly equivalence
tests.

The runtime cannot import `training.ocr.detector_inference` - that module reaches
`CamVidModel`, and so torch and pytorch_lightning, neither of which the runtime ships.
So the geometry exists twice, and the only thing keeping the copies honest is this file.
A drift in tile stride, patch padding or class order would not raise anywhere: it would
relabel or misplace boxes quietly, and the whole point of matching the reference is that
a predicted box and a ground-truth box stay comparable by construction.

Hence: every shared function is asserted equal to the reference on the same inputs,
including the awkward sizes (pages that are not a multiple of the stride, pages smaller
than one patch), rather than merely asserted self-consistent.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from homr import text_detection

try:  # The reference implementation is training-side and torch-only.
    import torch
    from torch import nn
    from torch.export import Dim

    from training.ocr import detector_inference as reference
    from training.ocr.detector_masks import CLASS_INDEX as REFERENCE_CLASS_INDEX
    from training.ocr.detector_masks import CLASS_ORDER as REFERENCE_CLASS_ORDER
    from training.ocr.detector_patches import PATCH_SIZE as REFERENCE_PATCH_SIZE
    from training.ocr.detector_patches import extract_patch as reference_extract_patch

    HAS_REFERENCE = True
except ImportError:  # pragma: no cover - exercised only in a runtime-only environment
    HAS_REFERENCE = False


needs_reference = unittest.skipUnless(
    HAS_REFERENCE, "training-side reference needs torch; equivalence unverifiable without it"
)


@needs_reference
class MatchesTheTrainingReference(unittest.TestCase):
    #: Deliberately includes sizes that are not multiples of the 160px stride, a page
    #: smaller than one patch in each dimension, and an exact multiple - the three cases
    #: where an off-by-one in `tile_origins` produces a silently uncovered strip.
    SHAPES = ((1849, 1007), (320, 320), (100, 5000), (5000, 100), (640, 480), (1, 1))

    def test_tile_origins_agree_on_every_awkward_page_shape(self):
        for height, width in self.SHAPES:
            with self.subTest(shape=(height, width)):
                self.assertEqual(
                    text_detection.tile_origins(height, width),
                    reference.tile_origins(height, width),
                )

    def test_tiles_cover_every_pixel(self):
        """The property the far-edge special case exists to guarantee."""
        for height, width in self.SHAPES:
            with self.subTest(shape=(height, width)):
                covered = np.zeros((height, width), dtype=bool)
                for y, x in text_detection.tile_origins(height, width):
                    covered[y : y + text_detection.PATCH_SIZE, x : x + text_detection.PATCH_SIZE] = True
                self.assertTrue(covered.all(), "a strip of the page is never predicted")

    def test_patch_size_and_stride_agree(self):
        self.assertEqual(text_detection.PATCH_SIZE, REFERENCE_PATCH_SIZE)
        self.assertEqual(text_detection.STEP, reference.STEP)

    def test_extract_patch_agrees_including_past_the_page_edge(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(400, 300, 3), dtype=np.uint8)
        # The last two fall off the bottom/right, which is where padding is applied.
        for origin in ((0, 0), (160, 160), (240, 0), (0, 160), (300, 200)):
            with self.subTest(origin=origin):
                np.testing.assert_array_equal(
                    text_detection.extract_patch(image, origin, 255),
                    reference_extract_patch(image, origin, 255),
                )

    def test_class_order_and_indices_agree(self):
        """The exported graph's channel order *is* this list; a divergence relabels
        every box rather than failing."""
        self.assertEqual(text_detection.CLASS_ORDER, REFERENCE_CLASS_ORDER)
        self.assertEqual(text_detection.CLASS_INDEX, REFERENCE_CLASS_INDEX)
        self.assertEqual(text_detection.NUM_CLASSES, reference.NUM_CLASSES)
        self.assertEqual(text_detection.BACKGROUND, 0)

    def test_softmax_agrees_with_torch(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(scale=8.0, size=(2, text_detection.NUM_CLASSES, 16, 16))
        np.testing.assert_allclose(
            text_detection._softmax(logits.astype(np.float32), axis=1),
            torch.from_numpy(logits).softmax(dim=1).numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_softmax_survives_logits_that_would_overflow_a_naive_exp(self):
        extreme = np.full((1, text_detection.NUM_CLASSES, 2, 2), 1000.0, dtype=np.float32)
        extreme[0, 3] = 1200.0
        probs = text_detection._softmax(extreme, axis=1)
        self.assertTrue(np.isfinite(probs).all())
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-6)
        self.assertEqual(int(probs[0, :, 0, 0].argmax()), 3)

    def test_box_recovery_agrees_on_a_synthetic_probability_map(self):
        probs = np.zeros((text_detection.NUM_CLASSES, 60, 80), dtype=np.float32)
        probs[text_detection.BACKGROUND] = 0.9
        # Two separated Lyrics blobs, one Tempo blob, and a 2px speck below min_area.
        probs[text_detection.CLASS_INDEX["Lyrics"], 5:15, 5:25] = 0.99
        probs[text_detection.CLASS_INDEX["Lyrics"], 30:40, 50:70] = 0.75
        probs[text_detection.CLASS_INDEX["Tempo"], 2:6, 40:60] = 0.95
        probs[text_detection.CLASS_INDEX["Fingering"], 50:51, 1:3] = 0.99

        mine = text_detection.boxes_from_probs(probs)
        theirs = reference.boxes_from_probs(probs)

        self.assertEqual(
            sorted((b.label, b.left, b.top, b.right, b.bottom) for b in mine),
            sorted((b.label, b.left, b.top, b.right, b.bottom) for b in theirs),
        )
        for mine_box, their_box in zip(
            sorted(mine, key=lambda b: (b.label, b.top, b.left)),
            sorted(theirs, key=lambda b: (b.label, b.top, b.left)),
            strict=True,
        ):
            self.assertAlmostEqual(mine_box.confidence, their_box.confidence, places=6)

    def test_the_min_area_speck_is_dropped_by_both(self):
        """Guards the equivalence test above from passing vacuously: if neither
        implementation dropped anything, the shared threshold would be untested."""
        probs = np.zeros((text_detection.NUM_CLASSES, 20, 20), dtype=np.float32)
        probs[text_detection.BACKGROUND] = 0.9
        probs[text_detection.CLASS_INDEX["Fingering"], 5:6, 5:7] = 0.99  # area 2 < 4
        self.assertEqual(text_detection.boxes_from_probs(probs), [])
        probs[text_detection.CLASS_INDEX["Fingering"], 5:7, 5:7] = 0.99  # area 4
        self.assertEqual(len(text_detection.boxes_from_probs(probs)), 1)


class ConstantClassModel(nn.Module if HAS_REFERENCE else object):  # type: ignore[misc]
    """Emits a fixed class over a fixed region, so the expected boxes are known exactly
    without depending on any trained checkpoint."""

    def __init__(self, class_index: int, num_classes: int) -> None:
        super().__init__()
        self.class_index = class_index
        self.num_classes = num_classes

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        batch = x.shape[0]
        logits = torch.zeros(batch, self.num_classes, 320, 320)
        logits[:, text_detection.BACKGROUND] = 1.0
        # A band in the middle of every tile, safely inside the patch.
        logits[:, self.class_index, 100:150, 60:200] = 9.0
        return logits


class ChannelSensitiveModel(nn.Module if HAS_REFERENCE else object):  # type: ignore[misc]
    """Fires where the *first* input channel dominates the third.

    `ConstantClassModel` ignores its input entirely, so it cannot notice a channel
    order or scaling mistake - a BGR/RGB flip passed every test against it. The
    reference reads pages with `cv2.imread` and never converts colour, so the model was
    trained on BGR; a runtime that helpfully converted to RGB would feed a silently
    different input distribution and still look correct.
    """

    def __init__(self, class_index: int, num_classes: int) -> None:
        super().__init__()
        self.class_index = class_index
        self.num_classes = num_classes

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        logits = torch.zeros(x.shape[0], self.num_classes, 320, 320)
        logits[:, text_detection.BACKGROUND] = 1.0
        dominant = (x[:, 0] - x[:, 2] > 0.1).float()
        logits[:, self.class_index] = 9.0 * dominant
        return logits


class ScaleSensitiveModel(nn.Module if HAS_REFERENCE else object):  # type: ignore[misc]
    """Fires only on inputs already scaled into [0, 1].

    The reference divides tiles by 255 before inference. Skipping that would leave every
    pixel two orders of magnitude too large, which no constant-output model can detect.
    """

    def __init__(self, class_index: int, num_classes: int) -> None:
        super().__init__()
        self.class_index = class_index
        self.num_classes = num_classes

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        logits = torch.zeros(x.shape[0], self.num_classes, 320, 320)
        logits[:, text_detection.BACKGROUND] = 1.0
        in_unit_range = (x[:, 0] < 0.9).float()
        logits[:, self.class_index] = 9.0 * in_unit_range
        return logits


def export_model(model: "nn.Module", path: Path) -> str:
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 320, 320),
        str(path),
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"x": (Dim("batch_size"), 3, 320, 320)},
        dynamo=True,
    )
    return str(path)


def export_constant_model(label: str, path: Path) -> str:
    """A one-class ONNX graph at `path`, exported the way `training/onnx/convert.py`
    exports the real detectors - eval mode, opset 18, `input`/`output` names - so the
    plumbing under test is the plumbing production uses."""
    return export_model(
        ConstantClassModel(text_detection.CLASS_INDEX[label], text_detection.NUM_CLASSES),
        path,
    )


@needs_reference
class RunsAnActualOnnxGraphOverAPage(unittest.TestCase):
    """The equivalence tests above never execute a graph. This one does, so that a
    mistake in the ONNX plumbing - input layout, normalisation, output name, the missing
    softmax - cannot pass unnoticed behind agreeing pure-numpy helpers."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = export_constant_model(
            "Lyrics", Path(self.directory.name) / "detector.onnx"
        )

    def test_detects_the_band_the_graph_emits(self):
        detector = text_detection.TextDetector(self.model_path)
        page = np.full((320, 320, 3), 255, dtype=np.uint8)
        boxes = detector.detect(page)

        self.assertEqual([box.label for box in boxes], ["Lyrics"])
        box = boxes[0]
        self.assertEqual((box.left, box.top, box.right, box.bottom), (60, 100, 200, 150))
        self.assertGreater(box.confidence, 0.9)

    def test_probabilities_are_normalised_after_stitching(self):
        """Overlap averaging divides by coverage; if that were wrong, probabilities in
        the doubly-predicted interior would not sum to one."""
        detector = text_detection.TextDetector(self.model_path)
        probs = detector.predict_mask(np.full((480, 480, 3), 255, dtype=np.uint8))
        np.testing.assert_allclose(probs.sum(axis=0), 1.0, rtol=1e-4, atol=1e-4)

    def test_a_page_larger_than_one_tile_is_fully_covered(self):
        detector = text_detection.TextDetector(self.model_path)
        probs = detector.predict_mask(np.full((700, 500, 3), 255, dtype=np.uint8))
        self.assertEqual(probs.shape, (text_detection.NUM_CLASSES, 700, 500))
        # No pixel was left at the zero-initialised default.
        self.assertTrue((probs.sum(axis=0) > 0.5).all())

    def test_emits_labels_it_is_not_allowed_to_contribute(self):
        """Class gating is the fusion policy's job, versioned with the calibration it
        belongs to. This module must not filter as well, or the same rule lives in two
        places and they drift."""
        path = export_constant_model(
            "Fingering", Path(self.directory.name) / "fingering.onnx"
        )
        boxes = text_detection.TextDetector(path).detect(
            np.full((320, 320, 3), 255, dtype=np.uint8)
        )
        self.assertEqual([box.label for box in boxes], ["Fingering"])

    def test_channel_order_reaches_the_graph_as_bgr(self):
        """A page whose region is blue-dominant in BGR. If the runtime flipped to RGB,
        the same pixels would read as red-dominant and the box would vanish."""
        path = export_model(
            ChannelSensitiveModel(
                text_detection.CLASS_INDEX["Lyrics"], text_detection.NUM_CLASSES
            ),
            Path(self.directory.name) / "channels.onnx",
        )
        page = np.zeros((320, 320, 3), dtype=np.uint8)
        page[..., 0] = 200  # blue channel in OpenCV's BGR order
        page[..., 2] = 10
        boxes = text_detection.TextDetector(path).detect(page)
        self.assertEqual([box.label for box in boxes], ["Lyrics"])
        self.assertEqual(
            (boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom), (0, 0, 320, 320)
        )

    def test_tiles_are_scaled_into_unit_range_before_inference(self):
        """Fires only below 0.9, so an unscaled tile (0-255) produces nothing."""
        path = export_model(
            ScaleSensitiveModel(
                text_detection.CLASS_INDEX["Tempo"], text_detection.NUM_CLASSES
            ),
            Path(self.directory.name) / "scale.onnx",
        )
        page = np.full((320, 320, 3), 200, dtype=np.uint8)  # 200/255 = 0.784 < 0.9
        boxes = text_detection.TextDetector(path).detect(page)
        self.assertEqual([box.label for box in boxes], ["Tempo"])

    def test_batching_does_not_change_the_result(self):
        detector = text_detection.TextDetector(self.model_path)
        page = np.full((700, 500, 3), 255, dtype=np.uint8)
        np.testing.assert_allclose(
            detector.predict_mask(page, batch_size=1),
            detector.predict_mask(page, batch_size=16),
            rtol=1e-5,
            atol=1e-6,
        )


class CachingAndFailure(unittest.TestCase):
    def setUp(self):
        text_detection._detector_cache.clear()
        self.addCleanup(text_detection._detector_cache.clear)

    def test_missing_weights_say_which_file_and_that_it_is_optional(self):
        with self.assertRaises(FileNotFoundError) as caught:
            text_detection.TextDetector("/nonexistent/detector.onnx")
        self.assertIn("/nonexistent/detector.onnx", str(caught.exception))
        self.assertIn("optional", str(caught.exception))

    @needs_reference
    def test_both_detectors_stay_resident_together(self):
        """Fusion runs both over the identical page, so neither may evict the other."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        paths = [
            export_constant_model(label, Path(directory.name) / name)
            for name, label in (("a.onnx", "Lyrics"), ("b.onnx", "Tempo"))
        ]

        first, second = (text_detection.get_detector(p) for p in paths)
        self.assertIsNot(first, second)
        self.assertIs(text_detection.get_detector(paths[0]), first)
        self.assertIs(text_detection.get_detector(paths[1]), second)
        self.assertEqual(len(text_detection._detector_cache), 2)


if __name__ == "__main__":
    unittest.main()
