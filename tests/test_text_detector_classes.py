"""Model labels are an ordered checkpoint contract, not a process-wide default."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from homr import text_detector_config
from homr.text_detector_classes import (
    DIRECTION_CLASS_ORDER,
    LEGACY_CLASS_ORDER,
    read_class_order,
    write_class_order,
)
from training.ocr.detector_inference import boxes_from_probs


class TestClassOrderMetadata(unittest.TestCase):
    def test_sidecar_round_trip_and_conflicting_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "detector.pth"
            write_class_order(weights, DIRECTION_CLASS_ORDER)
            self.assertEqual(read_class_order(weights), DIRECTION_CLASS_ORDER)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                read_class_order(weights, LEGACY_CLASS_ORDER)

    def test_unknown_checkpoint_needs_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "old.pth"
            with self.assertRaisesRegex(ValueError, "missing"):
                read_class_order(weights)
            self.assertEqual(read_class_order(weights, LEGACY_CLASS_ORDER), LEGACY_CLASS_ORDER)

    def test_pinned_released_onnx_models_keep_legacy_order(self) -> None:
        self.assertEqual(
            read_class_order(text_detector_config.detector_vocal_path), LEGACY_CLASS_ORDER
        )
        self.assertEqual(
            read_class_order(text_detector_config.detector_instrumental_path),
            LEGACY_CLASS_ORDER,
        )

    def test_same_channel_has_different_meaning_in_two_models(self) -> None:
        legacy = np.zeros((8, 24, 24), dtype=np.float32)
        legacy[3, 2:22, 2:22] = 1.0
        direction = legacy[:6]
        self.assertEqual(
            [box.label for box in boxes_from_probs(legacy, class_order=LEGACY_CLASS_ORDER)],
            ["Expression"],
        )
        self.assertEqual(
            [box.label for box in boxes_from_probs(direction, class_order=DIRECTION_CLASS_ORDER)],
            ["DirectionText"],
        )


if __name__ == "__main__":
    unittest.main()
