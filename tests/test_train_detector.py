import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from training.ocr.detector_patches import Sample
from training.ocr.train_detector import (
    CLASS_NAMES,
    NUM_CLASSES,
    collate,
    per_class_iou,
    train,
)


class TestClassNames(unittest.TestCase):
    def test_background_is_first(self) -> None:
        self.assertEqual(CLASS_NAMES[0], "background")

    def test_count_matches_class_order_plus_background(self) -> None:
        self.assertEqual(len(CLASS_NAMES), NUM_CLASSES)
        self.assertEqual(len(set(CLASS_NAMES)), NUM_CLASSES)


class TestCollate(unittest.TestCase):
    def test_images_are_normalised_to_zero_one_and_channels_first(self) -> None:
        batch = [
            (np.full((8, 8, 3), 255, dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8)),
        ]

        out = collate(batch)

        self.assertEqual(out["images"].shape, (1, 3, 8, 8))
        self.assertAlmostEqual(out["images"].max().item(), 1.0, places=5)

    def test_masks_stay_integer_class_indices(self) -> None:
        batch = [(np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8), 3, dtype=np.uint8))]

        out = collate(batch)

        self.assertEqual(out["masks"].dtype, torch.long)
        self.assertTrue((out["masks"] == 3).all())

    def test_a_batch_of_several_stacks_correctly(self) -> None:
        batch = [
            (np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))
            for _ in range(3)
        ]

        out = collate(batch)

        self.assertEqual(out["images"].shape[0], 3)
        self.assertEqual(out["masks"].shape[0], 3)


class TestPerClassIou(unittest.TestCase):
    class _StubModel:
        """Enough of CamVidModel's shape for per_class_iou - it only reads NUM_CLASSES."""

    def test_a_perfect_prediction_gives_iou_one_for_the_classes_present(self) -> None:
        # Two classes present: background (0) and Lyrics (index of Lyrics in CLASS_NAMES).
        lyrics_index = CLASS_NAMES.index("Lyrics")
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        masks[0, 0, 0] = lyrics_index
        logits = torch.full((1, NUM_CLASSES, 4, 4), -10.0)
        logits[0, 0] += 20.0
        logits[0, lyrics_index, 0, 0] += 40.0  # override background's edge at this pixel

        result = per_class_iou(self._StubModel(), logits, masks)

        self.assertAlmostEqual(result["background"], 1.0, places=3)
        self.assertAlmostEqual(result["Lyrics"], 1.0, places=3)

    def test_a_class_absent_from_the_batch_is_not_reported(self) -> None:
        # "No data" and "zero IoU" have to read differently, or a starved class and an
        # unlucky batch look identical.
        masks = torch.zeros((1, 4, 4), dtype=torch.long)
        logits = torch.zeros((1, NUM_CLASSES, 4, 4))

        result = per_class_iou(self._StubModel(), logits, masks)

        self.assertNotIn("Fingering", result)
        self.assertIn("background", result)


def _write_sample(directory: Path, name: str) -> Sample:
    image_path = directory / f"{name}.png"
    mask_path = directory / f"{name}.mask.png"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 255, dtype=np.uint8))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:30] = 1  # class index 1 = SystemText, first in CLASS_ORDER
    cv2.imwrite(str(mask_path), mask)
    return Sample(str(image_path), str(mask_path))


class TestTrainEntryPoint(unittest.TestCase):
    """The pattern that has caught defects everywhere else in this project: run the whole
    path for real, with a small stand-in, rather than trust units in isolation."""

    def _args(self, directory: Path, **overrides) -> Namespace:
        index = directory / "index.txt"
        samples = [_write_sample(directory, f"s{i}") for i in range(2)]
        index.write_text(
            "\n".join(f"{s.image},{s.mask}" for s in samples) + "\n", encoding="utf-8"
        )
        defaults = {
            "index": index,
            "weights": directory / "w.pth",
            "out": directory / "history.json",
            "patches_per_image": 2,
            "epochs": 1,
            "lr": 1e-4,
            "batch_size": 2,
            "workers": 0,
            "seed": 0,
            "skip_pretrained": True,
            "device": "cpu",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_the_whole_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = train(self._args(Path(tmp)))

        self.assertEqual(len(report["history"]), 1)
        self.assertIn("background", report["history"][0])

    def test_weights_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            train(self._args(directory))

            self.assertTrue((directory / "w.pth").exists())


if __name__ == "__main__":
    unittest.main()
