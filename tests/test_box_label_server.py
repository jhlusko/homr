"""The labelling tool must write what the detector's data loader reads.

A review tool whose output needs a conversion step before it can be used is a review tool
whose output quietly rots. `detector_data.boxes_of` is the reader; these tests pin the
writer against it rather than against a description of it.
"""

import json
import tempfile
import unittest
from pathlib import Path

from training.ocr.box_label_server import CLASSES, _load, _record_path, _store
from training.ocr.detector_data import boxes_of


class TestTheRecordShape(unittest.TestCase):
    def _write(self, boxes: list[dict]) -> tuple[Path, dict]:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        page = directory / "pages" / "IMSLP999" / "IMSLP999-p001.png"
        page.parent.mkdir(parents=True)
        page.write_bytes(b"")
        record = _store(directory / "out", page, boxes)
        return record, json.loads(record.read_text(encoding="utf-8"))

    def test_the_path_matches_the_existing_ground_truth(self) -> None:
        """`<score>/<score>:<page>.boxes.json`, which is what `collect` globs for."""
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        page = directory / "IMSLP999" / "IMSLP999-p007.png"
        self.assertEqual(
            "IMSLP999/IMSLP999:0007.boxes.json",
            str(_record_path(directory / "out", page).relative_to(directory / "out")),
        )

    def test_the_page_is_reachable_from_the_record(self) -> None:
        """`collect` resolves `image` against the record's own directory."""
        record, payload = self._write([])
        self.assertTrue((record.parent / payload["image"]).exists())

    def test_the_reader_recovers_every_box(self) -> None:
        record, _payload = self._write(
            [
                {"label": "Tempo", "left": 10, "top": 20, "right": 90, "bottom": 40},
                {"label": "Dynamic", "left": 5, "top": 60, "right": 25, "bottom": 75},
            ]
        )
        recovered = boxes_of(json.loads(record.read_text(encoding="utf-8")), str(record))
        self.assertEqual({"Tempo", "Dynamic"}, {box.label for box in recovered})

    def test_lyrics_is_written_empty_rather_than_omitted(self) -> None:
        """`boxes_of` reads that key directly, so the shape must never vary."""
        _record, payload = self._write([])
        self.assertEqual([], payload["lyrics"])
        self.assertEqual({}, payload["text_boxes"])

    def test_an_unknown_class_is_dropped_not_written(self) -> None:
        _record, payload = self._write(
            [{"label": "Fingering", "left": 1, "top": 1, "right": 2, "bottom": 2}]
        )
        self.assertEqual({}, payload["text_boxes"])

    def test_coordinates_are_written_as_integers(self) -> None:
        _record, payload = self._write(
            [{"label": "Tempo", "left": 1.6, "top": 2.4, "right": 9.5, "bottom": 8.2}]
        )
        box = payload["text_boxes"]["Tempo"][0]
        self.assertTrue(all(isinstance(value, int) for value in box.values()), box)


class TestReopening(unittest.TestCase):
    def test_a_saved_page_comes_back_for_correction(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        page = directory / "IMSLP999" / "IMSLP999-p001.png"
        page.parent.mkdir(parents=True)
        boxes = [{"label": "Expression", "left": 3, "top": 4, "right": 30, "bottom": 14}]
        _store(directory / "out", page, boxes)
        self.assertEqual(boxes, _load(directory / "out", page))

    def test_an_unlabelled_page_starts_empty(self) -> None:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.assertEqual([], _load(directory / "out", directory / "X" / "X-p001.png"))


class TestTheClassList(unittest.TestCase):
    def test_lyrics_is_not_offered(self) -> None:
        """A separate detector serves lyrics; mixing them is what fusion_policy avoids."""
        self.assertNotIn("Lyrics", CLASSES)

    def test_every_class_is_one_the_detector_knows(self) -> None:
        from training.ocr.detector_masks import CLASS_INDEX

        for name in CLASSES:
            self.assertIn(name, CLASS_INDEX, name)


if __name__ == "__main__":
    unittest.main()


class TestPageSelection(unittest.TestCase):
    """One page per score, and it has to be a page with music on it.

    76 of 215 Lieder scores open with a title page. Serving `*-p001.png` blindly handed
    back a blank sheet 39% of the time in the first labelling run.
    """

    def _build(self) -> tuple[Path, Path]:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        pages, systems = directory / "pages", directory / "systems"
        (pages / "IMSLP1").mkdir(parents=True)
        (pages / "IMSLP2").mkdir(parents=True)
        systems.mkdir()
        for name in ("IMSLP1-p001.png", "IMSLP1-p002.png"):
            (pages / "IMSLP1" / name).write_bytes(b"")
        (pages / "IMSLP2" / "IMSLP2-p001.png").write_bytes(b"")
        # IMSLP1 opens with a title page; IMSLP2 starts straight into music.
        (systems / "IMSLP1.yaml").write_text(
            "pages:\n"
            "  1: {image: IMSLP1/IMSLP1-p001.png, systems: []}\n"
            "  2: {image: IMSLP1/IMSLP1-p002.png, systems: [{boundingBox: {top: 1}}]}\n"
        )
        (systems / "IMSLP2.yaml").write_text(
            "pages:\n  1: {image: IMSLP2/IMSLP2-p001.png, systems: [{boundingBox: {top: 1}}]}\n"
        )
        return pages, systems

    def test_a_title_page_is_skipped_for_the_first_page_with_music(self) -> None:
        from training.ocr.box_label_server import _pages

        pages, systems = self._build()
        self.assertEqual(
            ["IMSLP1-p002.png", "IMSLP2-p001.png"],
            [p.name for p in _pages(pages, 0, systems)],
        )

    def test_one_page_per_score(self) -> None:
        from training.ocr.box_label_server import _pages

        pages, systems = self._build()
        chosen = _pages(pages, 0, systems)
        self.assertEqual(len(chosen), len({p.parent.name for p in chosen}))

    def test_without_detection_it_falls_back_to_page_one(self) -> None:
        from training.ocr.box_label_server import _pages

        pages, _systems = self._build()
        self.assertEqual(
            ["IMSLP1-p001.png", "IMSLP2-p001.png"], [p.name for p in _pages(pages, 0, None)]
        )
