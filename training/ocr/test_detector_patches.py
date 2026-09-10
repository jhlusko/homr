import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.ocr.detector_patches import (
    PAD_IMAGE_VALUE,
    PAD_MASK_VALUE,
    PATCH_SIZE,
    DetectorPatches,
    Sample,
    box_centres_by_class,
    extract_patch,
    patch_origin,
    read_index,
)


def _mask_with_boxes(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for top, left, bottom, right in boxes:
        mask[top:bottom, left:right] = 1
    return mask


class TestReadIndex(unittest.TestCase):
    def test_lines_become_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.txt"
            path.write_text("a.png,a.mask.png\nb.png,b.mask.png\n", encoding="utf-8")

            samples = read_index(path)

        self.assertEqual(samples, [Sample("a.png", "a.mask.png"), Sample("b.png", "b.mask.png")])

    def test_blank_lines_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.txt"
            path.write_text("a.png,a.mask.png\n\n", encoding="utf-8")

            self.assertEqual(len(read_index(path)), 1)


class TestBoxCentresByClass(unittest.TestCase):
    """`_mask_with_boxes` writes class value 1, which `CLASS_INDEX` assigns to
    "SystemText" (the first entry in `CLASS_ORDER`) - these boxes land there.
    """

    def test_one_box_gives_one_centre(self) -> None:
        mask = _mask_with_boxes((100, 100), [(10, 10, 30, 50)])

        centres = box_centres_by_class(mask)["SystemText"]

        self.assertEqual(len(centres), 1)
        y, x = centres[0]
        self.assertTrue(10 <= y <= 30 and 10 <= x <= 50)

    def test_two_separate_boxes_give_two_centres(self) -> None:
        mask = _mask_with_boxes((100, 100), [(0, 0, 10, 10), (80, 80, 90, 90)])

        self.assertEqual(len(box_centres_by_class(mask)["SystemText"]), 2)

    def test_an_empty_mask_has_no_classes(self) -> None:
        self.assertEqual(box_centres_by_class(np.zeros((50, 50), dtype=np.uint8)), {})

    def test_different_classes_are_kept_separate(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1  # SystemText
        mask[60:80, 60:80] = 2  # Fingering

        by_class = box_centres_by_class(mask)

        self.assertEqual(set(by_class), {"SystemText", "Fingering"})
        self.assertEqual(len(by_class["SystemText"]), 1)
        self.assertEqual(len(by_class["Fingering"]), 1)


class TestPatchOrigin(unittest.TestCase):
    def test_the_origin_stays_within_the_page(self) -> None:
        import random

        origin = patch_origin((5, 5), (1000, 1000), jitter=0.3, rng=random.Random(0))

        self.assertGreaterEqual(origin[0], 0)
        self.assertGreaterEqual(origin[1], 0)

    def test_the_origin_never_exceeds_what_a_full_patch_needs(self) -> None:
        import random

        origin = patch_origin((999, 999), (1000, 1000), jitter=0.3, rng=random.Random(0))

        self.assertLessEqual(origin[0], 1000 - PATCH_SIZE)
        self.assertLessEqual(origin[1], 1000 - PATCH_SIZE)

    def test_a_page_smaller_than_one_patch_still_returns_a_valid_origin(self) -> None:
        import random

        origin = patch_origin((10, 10), (50, 50), jitter=0.3, rng=random.Random(0))

        self.assertEqual(origin, (0, 0))


class TestExtractPatch(unittest.TestCase):
    def test_a_patch_fully_inside_the_page_has_no_padding(self) -> None:
        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        patch = extract_patch(image, (50, 50), PAD_IMAGE_VALUE)

        self.assertTrue((patch == image[50 : 50 + PATCH_SIZE, 50 : 50 + PATCH_SIZE]).all())

    def test_a_patch_past_the_edge_is_padded_with_the_given_value(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        patch = extract_patch(image, (0, 0), 255)

        # Only the top-left 100x100 came from the page; the rest is padding.
        self.assertTrue((patch[100:, :] == 255).all())
        self.assertTrue((patch[:, 100:] == 255).all())

    def test_image_padding_is_paper_not_ink(self) -> None:
        # 27.51 found zero-padding teaches the model every crop ends in a black bar.
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        patch = extract_patch(image, (0, 0), PAD_IMAGE_VALUE)

        self.assertEqual(patch[PATCH_SIZE - 1, PATCH_SIZE - 1, 0], PAD_IMAGE_VALUE)

    def test_mask_padding_is_background(self) -> None:
        mask = np.full((50, 50), 7, dtype=np.uint8)

        patch = extract_patch(mask, (0, 0), PAD_MASK_VALUE)

        self.assertEqual(patch[PATCH_SIZE - 1, PATCH_SIZE - 1], PAD_MASK_VALUE)

    def test_the_patch_is_always_the_fixed_size(self) -> None:
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        patch = extract_patch(image, (0, 0), PAD_IMAGE_VALUE)

        self.assertEqual(patch.shape, (PATCH_SIZE, PATCH_SIZE, 3))


def _write_sample(directory: Path, name: str, size=(600, 600), boxes=None) -> Sample:
    image_path = directory / f"{name}.png"
    mask_path = directory / f"{name}.mask.png"
    cv2.imwrite(str(image_path), np.full((*size, 3), 255, dtype=np.uint8))
    mask = _mask_with_boxes(size, boxes or [(100, 100, 120, 150)])
    cv2.imwrite(str(mask_path), mask)
    return Sample(str(image_path), str(mask_path))


class TestDetectorPatches(unittest.TestCase):
    def test_the_length_is_samples_times_patches_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = [_write_sample(Path(tmp), "a"), _write_sample(Path(tmp), "b")]

            dataset = DetectorPatches(samples, patches_per_image=5)

        self.assertEqual(len(dataset), 10)

    def test_every_patch_has_the_fixed_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = [_write_sample(Path(tmp), "a")]
            dataset = DetectorPatches(samples, patches_per_image=3)

            image_patch, mask_patch = dataset[0]

        self.assertEqual(image_patch.shape, (PATCH_SIZE, PATCH_SIZE, 3))
        self.assertEqual(mask_patch.shape, (PATCH_SIZE, PATCH_SIZE))

    def test_positive_ratio_one_always_hits_a_box(self) -> None:
        # At the sampling rate this module exists to justify - 27.69 measured 0.38% page
        # coverage - a purely random sampler would rarely land on a box at all.
        with tempfile.TemporaryDirectory() as tmp:
            samples = [_write_sample(Path(tmp), "a")]
            dataset = DetectorPatches(samples, patches_per_image=10, positive_ratio=1.0)

            hits = sum(1 for i in range(10) if (dataset[i][1] != PAD_MASK_VALUE).any())

        self.assertEqual(hits, 10)

    def test_positive_ratio_zero_never_forces_a_box(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # A large page with a tiny box, so a uniformly random patch is unlikely to
            # land on it - this is a statistical check, not a guarantee, so it allows slack.
            samples = [_write_sample(Path(tmp), "a", size=(4000, 4000), boxes=[(0, 0, 5, 5)])]
            dataset = DetectorPatches(samples, patches_per_image=20, positive_ratio=0.0)

            hits = sum(1 for i in range(20) if (dataset[i][1] != PAD_MASK_VALUE).any())

        self.assertLess(hits, 20)

    def test_a_missing_file_raises_rather_than_returning_something_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = DetectorPatches([Sample("absent.png", "absent.mask.png")])

            with self.assertRaises(FileNotFoundError):
                dataset[0]

    def test_the_same_index_gives_the_same_patch(self) -> None:
        # A seeded rng, so a training run is reproducible run to run.
        with tempfile.TemporaryDirectory() as tmp:
            samples = [_write_sample(Path(tmp), "a")]

            first = DetectorPatches(samples, seed=0)[0]
            second = DetectorPatches(samples, seed=0)[0]

        self.assertTrue((first[0] == second[0]).all())


if __name__ == "__main__":
    unittest.main()
