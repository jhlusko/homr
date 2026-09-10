import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.ocr.detector_data import Box
from training.ocr.detector_masks import (
    BACKGROUND,
    CLASS_INDEX,
    CLASS_ORDER,
    describe,
    rasterize,
    write_index,
    write_masks,
)


class TestClassOrder(unittest.TestCase):
    def test_background_is_always_zero(self) -> None:
        self.assertEqual(BACKGROUND, 0)

    def test_every_class_has_a_distinct_positive_index(self) -> None:
        indices = list(CLASS_INDEX.values())

        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(all(i > 0 for i in indices))

    def test_lyrics_is_a_known_class(self) -> None:
        self.assertIn("Lyrics", CLASS_ORDER)


class TestRasterize(unittest.TestCase):
    def test_a_box_is_painted_with_its_class_index(self) -> None:
        boxes = [Box("img", "Lyrics", 2, 2, 6, 6)]

        mask = rasterize(10, 10, boxes)

        self.assertEqual(mask[3, 3], CLASS_INDEX["Lyrics"])
        self.assertEqual(mask[0, 0], BACKGROUND)

    def test_a_box_touching_the_edge_is_clipped_not_dropped(self) -> None:
        boxes = [Box("img", "Lyrics", -5, -5, 5, 5)]

        mask = rasterize(10, 10, boxes)

        self.assertEqual(mask[0, 0], CLASS_INDEX["Lyrics"])

    def test_a_box_entirely_outside_the_page_paints_nothing(self) -> None:
        boxes = [Box("img", "Lyrics", 100, 100, 110, 110)]

        mask = rasterize(10, 10, boxes)

        self.assertTrue((mask == BACKGROUND).all())

    def test_an_unknown_class_is_ignored_rather_than_crashing(self) -> None:
        boxes = [Box("img", "NotAClass", 0, 0, 5, 5)]

        mask = rasterize(10, 10, boxes)

        self.assertTrue((mask == BACKGROUND).all())

    def test_later_boxes_win_on_overlap(self) -> None:
        # Priority order, documented rather than left to whatever order a dict iterates in.
        boxes = [Box("img", "SystemText", 0, 0, 10, 10), Box("img", "Lyrics", 0, 0, 10, 10)]

        mask = rasterize(10, 10, boxes)

        self.assertTrue((mask == CLASS_INDEX["Lyrics"]).all())

    def test_system_text_is_folded_into_staff_text(self) -> None:
        # 27.92: SystemText stayed at 0% whole-page precision/recall even with real
        # synthetic training data, unlike Fingering, which the same technique fixed -
        # folded into StaffText rather than continuing to spend training attention on a
        # class this detector cannot resolve.
        self.assertNotIn("SystemText", CLASS_ORDER)
        boxes = [Box("img", "SystemText", 2, 2, 6, 6)]

        mask = rasterize(10, 10, boxes)

        self.assertEqual(mask[3, 3], CLASS_INDEX["StaffText"])


class TestWriteMasks(unittest.TestCase):
    def test_a_mask_is_written_per_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            system = directory / "s1"
            system.mkdir()
            image_path = system / "s1-1.png"
            cv2.imwrite(str(image_path), np.full((20, 20, 3), 255, dtype=np.uint8))
            record = {
                "image": "s1-1.png", "width": 20, "height": 20,
                "lyrics": [{"text": "x", "left": 2, "top": 2, "right": 6, "bottom": 6}],
                "text_boxes": {},
            }
            import json
            (system / "s1.boxes.json").write_text(json.dumps(record), encoding="utf-8")

            pairs = write_masks(directory, directory / "out")

            self.assertEqual(len(pairs), 1)
            mask = cv2.imread(pairs[0][1], cv2.IMREAD_GRAYSCALE)
            self.assertEqual(mask[3, 3], CLASS_INDEX["Lyrics"])

    def test_an_image_that_cannot_be_read_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            system = directory / "s1"
            system.mkdir()
            record = {
                "image": "missing.png", "width": 20, "height": 20,
                "lyrics": [{"text": "x", "left": 0, "top": 0, "right": 5, "bottom": 5}],
                "text_boxes": {},
            }
            import json
            (system / "s1.boxes.json").write_text(json.dumps(record), encoding="utf-8")

            pairs = write_masks(directory, directory / "out")

        self.assertEqual(pairs, [])


class TestWriteIndex(unittest.TestCase):
    def test_pairs_round_trip_as_csv_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.txt"
            write_index([("a.png", "a.mask.png")], path)

            self.assertEqual(path.read_text(encoding="utf-8").strip(), "a.png,a.mask.png")


class TestDescribe(unittest.TestCase):
    def test_no_pairs_is_reported_not_crashed(self) -> None:
        self.assertIn("no masks", describe([]))

    def test_coverage_is_reported_for_real_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            mask_path = directory / "m.png"
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[0:2, 0:2] = 1
            cv2.imwrite(str(mask_path), mask)

            report = describe([("img.png", str(mask_path))])

        self.assertIn("text pixels", report)


if __name__ == "__main__":
    unittest.main()
