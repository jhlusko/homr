import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.ocr.detector_masks import CLASS_INDEX
from training.ocr.detector_patches import PATCH_SIZE, PreExtractedPatches, Sample, read_index
from training.ocr.extract_patch_bank import _draw_one, seed_for_image


def _write_page(directory: Path, name: str, size=(1200, 900)) -> Sample:
    image_path = directory / f"{name}.png"
    mask_path = directory / f"{name}.mask.png"
    cv2.imwrite(str(image_path), np.full((*size, 3), 255, dtype=np.uint8))
    mask = np.zeros(size, dtype=np.uint8)
    mask[100:140, 100:180] = CLASS_INDEX["Lyrics"]
    cv2.imwrite(str(mask_path), mask)
    return Sample(str(image_path), str(mask_path))


class TestSeedForImage(unittest.TestCase):
    def test_the_same_image_and_seed_always_give_the_same_value(self) -> None:
        self.assertEqual(seed_for_image(7, 12), seed_for_image(7, 12))

    def test_different_images_get_different_seeds(self) -> None:
        self.assertNotEqual(seed_for_image(7, 12), seed_for_image(7, 13))

    def test_different_runs_get_different_seeds(self) -> None:
        self.assertNotEqual(seed_for_image(7, 12), seed_for_image(8, 12))


class TestDrawOne(unittest.TestCase):
    def test_writes_one_file_pair_per_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            sample = _write_page(root, "a")

            written = _draw_one((0, sample, 5, 0.7, 0, out))

        self.assertEqual(len(written), 5)

    def test_every_written_patch_is_patch_sized(self) -> None:
        # The whole point of the bank is that training reads these directly, so a
        # wrong-sized file would only surface as a collate error mid-run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            sample = _write_page(root, "a")

            written = _draw_one((0, sample, 3, 1.0, 0, out))

            for image_path, mask_path in written:
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                self.assertEqual(image.shape, (PATCH_SIZE, PATCH_SIZE, 3))
                self.assertEqual(mask.shape, (PATCH_SIZE, PATCH_SIZE))

    def test_the_bank_is_reproducible_across_runs(self) -> None:
        # A bank that differed run to run would break the comparison the experiment
        # matrix rests on: E0-E3 must differ by their training setup, not by having
        # been handed different patches.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = _write_page(root, "a")
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()

            first = _draw_one((0, sample, 4, 0.7, 0, first_dir))
            second = _draw_one((0, sample, 4, 0.7, 0, second_dir))

            for (image_a, mask_a), (image_b, mask_b) in zip(first, second, strict=True):
                self.assertTrue((cv2.imread(image_a) == cv2.imread(image_b)).all())
                self.assertTrue((cv2.imread(mask_a) == cv2.imread(mask_b)).all())

    def test_a_positive_ratio_of_one_keeps_the_boxes_in_the_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            sample = _write_page(root, "a")

            written = _draw_one((0, sample, 4, 1.0, 0, out))
            hits = sum(
                1 for _, mask in written if (cv2.imread(mask, cv2.IMREAD_GRAYSCALE) != 0).any()
            )

        self.assertEqual(hits, 4)

    def test_patches_from_different_images_do_not_collide_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            first = _draw_one((0, _write_page(root, "a"), 3, 0.7, 0, out))
            second = _draw_one((1, _write_page(root, "b"), 3, 0.7, 0, out))

        self.assertEqual(len({p for p, _ in first} & {p for p, _ in second}), 0)


class TestPreExtractedPatches(unittest.TestCase):
    def test_reads_the_patch_back_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            written = _draw_one((0, _write_page(root, "a"), 2, 1.0, 0, out))
            dataset = PreExtractedPatches([Sample(i, m) for i, m in written])

            image, mask = dataset[0]

        self.assertEqual(image.shape, (PATCH_SIZE, PATCH_SIZE, 3))
        self.assertEqual(mask.shape, (PATCH_SIZE, PATCH_SIZE))

    def test_its_length_is_the_number_of_patches_not_pages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            written = _draw_one((0, _write_page(root, "a"), 6, 0.7, 0, out))
            dataset = PreExtractedPatches([Sample(i, m) for i, m in written])

        self.assertEqual(len(dataset), 6)

    def test_a_missing_patch_raises_rather_than_returning_something_wrong(self) -> None:
        dataset = PreExtractedPatches([Sample("absent.png", "absent.mask.png")])

        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_the_bank_index_round_trips_through_read_index(self) -> None:
        # extract_patch_bank writes the same "image,mask" format every other tool in
        # this pipeline reads, so the bank is usable wherever a page index was.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "patches"
            out.mkdir()
            written = _draw_one((0, _write_page(root, "a"), 2, 0.7, 0, out))
            index = root / "index.txt"
            index.write_text("\n".join(f"{i},{m}" for i, m in written) + "\n", encoding="utf-8")

            samples = read_index(index)

        self.assertEqual([(s.image, s.mask) for s in samples], written)


if __name__ == "__main__":
    unittest.main()
