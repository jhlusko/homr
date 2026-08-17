import json
import tempfile
import unittest
from pathlib import Path

from training.ocr.detector_data import (
    DETECTION_CLASSES,
    Box,
    boxes_of,
    collect,
    describe,
    write_manifest,
)


def _record(lyrics=None, text_boxes=None) -> dict:
    return {"image": "s-1.png", "width": 1000, "height": 500,
            "lyrics": lyrics or [], "text_boxes": text_boxes or {}}


class TestDynamicExcluded(unittest.TestCase):
    """27.45: Dynamic renders square where every text class is wide, because MusicXML
    stores it as an element name rather than a string - it is not text."""

    def test_dynamic_is_not_a_detection_class(self) -> None:
        self.assertNotIn("Dynamic", DETECTION_CLASSES)

    def test_a_dynamic_box_is_dropped_even_when_present(self) -> None:
        record = _record(text_boxes={"Dynamic": [{"left": 1, "top": 1, "right": 5, "bottom": 5}]})

        self.assertEqual(boxes_of(record, "img.png"), [])


class TestBoxesOf(unittest.TestCase):
    def test_lyrics_become_boxes_labelled_lyrics(self) -> None:
        record = _record(lyrics=[{"text": "va", "left": 1, "top": 1, "right": 5, "bottom": 5}])

        boxes = boxes_of(record, "img.png")

        self.assertEqual(boxes[0].label, "Lyrics")
        self.assertEqual((boxes[0].left, boxes[0].bottom), (1, 5))

    def test_other_text_classes_become_boxes_too(self) -> None:
        record = _record(text_boxes={"Tempo": [{"left": 1, "top": 1, "right": 5, "bottom": 5}]})

        boxes = boxes_of(record, "img.png")

        self.assertEqual(boxes[0].label, "Tempo")

    def test_a_class_outside_the_detection_set_is_dropped(self) -> None:
        record = _record(text_boxes={"NotARealClass": [{"left": 1, "top": 1, "right": 5, "bottom": 5}]})

        self.assertEqual(boxes_of(record, "img.png"), [])

    def test_every_box_carries_the_image_it_came_from(self) -> None:
        record = _record(lyrics=[{"text": "va", "left": 0, "top": 0, "right": 1, "bottom": 1}])

        self.assertEqual(boxes_of(record, "path/to/img.png")[0].image, "path/to/img.png")


class TestCollect(unittest.TestCase):
    def test_boxes_are_gathered_across_every_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("a", "b"):
                system = directory / name
                system.mkdir()
                (system / f"{name}.boxes.json").write_text(
                    json.dumps(_record(lyrics=[{"text": "x", "left": 0, "top": 0, "right": 1, "bottom": 1}])),
                    encoding="utf-8",
                )

            boxes = collect(directory)

        self.assertEqual(len(boxes), 2)

    def test_the_image_path_is_resolved_relative_to_the_record(self) -> None:
        # A detector loads images by path; a path relative to the wrong directory would
        # silently point at nothing once training starts.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            system = directory / "s1"
            system.mkdir()
            (system / "s1.boxes.json").write_text(
                json.dumps(_record(lyrics=[{"text": "x", "left": 0, "top": 0, "right": 1, "bottom": 1}])),
                encoding="utf-8",
            )

            boxes = collect(directory)

        self.assertEqual(boxes[0].image, str(system / "s-1.png"))


class TestManifestAndDescribe(unittest.TestCase):
    def test_the_manifest_round_trips(self) -> None:
        boxes = [Box("img.png", "Lyrics", 1, 2, 3, 4)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            write_manifest(boxes, path)
            row = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertEqual(row["label"], "Lyrics")
        self.assertEqual(row["right"], 3)

    def test_the_report_names_the_worst_imbalance(self) -> None:
        boxes = [Box("i", "Lyrics", 0, 0, 1, 1)] * 100 + [Box("i", "Tempo", 0, 0, 1, 1)]

        report = describe(boxes)

        self.assertIn("Lyrics is", report)
        self.assertIn("100x", report)

    def test_a_single_class_reports_without_dividing_by_zero(self) -> None:
        report = describe([Box("i", "Lyrics", 0, 0, 1, 1)])

        self.assertIn("1x", report)


if __name__ == "__main__":
    unittest.main()
