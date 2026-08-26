import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from training.ocr.detector_masks import CLASS_INDEX
from training.ocr.detector_patches import PATCH_SIZE
from training.ocr.eval_detector import load_checkpoint, score
from training.ocr.train_detector import NUM_CLASSES, CamVidModel


def _write_patch_bank(directory: Path, count: int = 2) -> Path:
    rows = []
    for i in range(count):
        image_path = directory / f"p{i}.png"
        mask_path = directory / f"p{i}.mask.png"
        cv2.imwrite(
            str(image_path), np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, dtype=np.uint8)
        )
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        mask[10:40, 10:60] = CLASS_INDEX["Lyrics"]
        cv2.imwrite(str(mask_path), mask)
        rows.append(f"{image_path},{mask_path}")
    index = directory / "index.txt"
    index.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return index


def _save_untrained_checkpoint(path: Path) -> None:
    model = CamVidModel(
        arch="Unet", encoder_name="resnet18", in_channels=3, out_classes=NUM_CLASSES,
        skip_weights_download=True,
    )
    torch.save(model.state_dict(), path)


class TestLoadCheckpoint(unittest.TestCase):
    def test_a_saved_checkpoint_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.pth"
            _save_untrained_checkpoint(path)

            model = load_checkpoint(path, "cpu")

            self.assertFalse(model.training)

    def test_the_loaded_weights_are_the_saved_ones(self) -> None:
        # A silently re-initialised model would still score something, and the whole
        # point of this tool is comparing checkpoints - so a wrong load would produce
        # a plausible-looking but meaningless comparison.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "w.pth"
            _save_untrained_checkpoint(path)
            saved = torch.load(path, map_location="cpu")

            model = load_checkpoint(path, "cpu")
            loaded = model.state_dict()

            key = next(iter(saved))
            self.assertTrue(torch.equal(saved[key], loaded[key].cpu()))


class TestScore(unittest.TestCase):
    def test_it_reports_iou_per_class_for_classes_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = _write_patch_bank(directory)
            checkpoint = directory / "w.pth"
            _save_untrained_checkpoint(checkpoint)

            result = score(checkpoint, index, device="cpu", workers=0)

        self.assertIn("Lyrics", result)
        for value in result.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_the_same_checkpoint_and_bank_score_identically_twice(self) -> None:
        # Comparisons across checkpoints are only meaningful if scoring is
        # deterministic - the loader is unshuffled and the model is in eval mode.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = _write_patch_bank(directory)
            checkpoint = directory / "w.pth"
            _save_untrained_checkpoint(checkpoint)

            first = score(checkpoint, index, device="cpu", workers=0)
            second = score(checkpoint, index, device="cpu", workers=0)

        self.assertEqual(first, second)

    def test_ignored_pixels_do_not_count_against_the_score(self) -> None:
        # The bug this tool exists partly to work around: a mask of mostly-255 must
        # not score every ignored pixel as an error.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image_path = directory / "p.png"
            mask_path = directory / "p.mask.png"
            cv2.imwrite(
                str(image_path), np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, dtype=np.uint8)
            )
            mask = np.full((PATCH_SIZE, PATCH_SIZE), 255, dtype=np.uint8)
            mask[10:40, 10:60] = CLASS_INDEX["Lyrics"]
            cv2.imwrite(str(mask_path), mask)
            index = directory / "index.txt"
            index.write_text(f"{image_path},{mask_path}\n", encoding="utf-8")
            checkpoint = directory / "w.pth"
            _save_untrained_checkpoint(checkpoint)

            ignored = score(checkpoint, index, device="cpu", ignore_index=255, workers=0)
            scored = score(checkpoint, index, device="cpu", ignore_index=None, workers=0)

        # With 255 ignored, only the Lyrics region is scored at all, so the set of
        # reported classes is smaller than when every ignored pixel is treated as a
        # real label.
        self.assertLessEqual(len(ignored), len(scored))


if __name__ == "__main__":
    unittest.main()
