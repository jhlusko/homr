import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.lyric_crops import (
    MARGIN,
    crop_syllables,
    describe,
    score_of,
    split_scores,
    write_manifest,
)


def _system(directory: Path, name: str, boxes: list[dict]) -> Path:
    page = np.full((200, 400, 3), 255, dtype=np.uint8)
    for box in boxes:
        page[box["top"]:box["bottom"], box["left"]:box["right"]] = 0
    cv2.imwrite(str(directory / f"{name}-1.png"), page)
    record = {"image": f"{name}-1.png", "width": 400, "height": 200, "dpi": 300,
              "lyrics": boxes, "text_boxes": {}, "extenders": []}
    path = directory / f"{name}.boxes.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _box(text: str, left: int, top: int = 50) -> dict:
    return {"text": text, "syllabic": "single", "verse": "1",
            "left": left, "top": top, "right": left + 30, "bottom": top + 20}


class TestScoreOf(unittest.TestCase):
    def test_the_score_is_everything_before_the_first_underscore(self) -> None:
        self.assertEqual(score_of("6007571_p1-s1"), "6007571")

    def test_a_name_without_a_system_suffix_is_its_own_score(self) -> None:
        self.assertEqual(score_of("6007571"), "6007571")


class TestSplitScores(unittest.TestCase):
    """A Lied's systems share its engraving, its typesetting and most of its words, so a
    crop-level split would measure memorisation and call it recognition."""

    def test_no_score_appears_on_both_sides(self) -> None:
        train, valid = split_scores([f"score{index}" for index in range(40)])

        self.assertEqual(train & valid, set())

    def test_every_score_is_placed(self) -> None:
        names = [f"score{index}" for index in range(40)]

        train, valid = split_scores(names)

        self.assertEqual(train | valid, set(names))

    def test_the_split_is_the_same_every_time(self) -> None:
        names = [f"score{index}" for index in range(40)]

        self.assertEqual(split_scores(names), split_scores(list(reversed(names))))

    def test_repeated_names_do_not_double_count(self) -> None:
        train, valid = split_scores(["a", "a", "b", "b", "c"])

        self.assertEqual(train | valid, {"a", "b", "c"})


class TestCropSyllables(unittest.TestCase):
    def test_one_crop_per_syllable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("va", 10), _box("gues", 60)])

            crops = crop_syllables(record, directory / "out")

        self.assertEqual([crop.text for crop in crops], ["va", "gues"])

    def test_the_crop_keeps_a_margin_of_page_around_it(self) -> None:
        # A hyphen sitting just outside the box is what separates a word's last syllable
        # from its middle one, so the crop is not cut exactly to the ink.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("va", 50)])

            crops = crop_syllables(record, directory / "out")
            image = cv2.imread(str(crops[0].path))

        self.assertEqual(image.shape[1], 30 + 2 * MARGIN)

    def test_a_box_at_the_page_edge_is_clipped_not_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("va", 0, top=0)])

            crops = crop_syllables(record, directory / "out")

        self.assertEqual(len(crops), 1)

    def test_the_score_travels_with_each_crop(self) -> None:
        # It is what the split is made on, so losing it here would lose the guarantee.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "6007571_p2-s3", [_box("va", 10)])

            crops = crop_syllables(record, directory / "out")

        self.assertEqual(crops[0].score, "6007571")

    def test_a_missing_page_yields_nothing_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("va", 10)])
            (directory / "s1_p1-s1-1.png").unlink()

            self.assertEqual(crop_syllables(record, directory / "out"), [])


class TestManifest(unittest.TestCase):
    def test_each_line_carries_the_crop_and_what_it_says(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("va", 10)])
            crops = crop_syllables(record, directory / "out")
            manifest = directory / "train.jsonl"

            write_manifest(crops, manifest)
            line = json.loads(manifest.read_text(encoding="utf-8").strip())

        self.assertEqual(line["text"], "va")
        self.assertEqual(line["score"], "s1")

    def test_non_ascii_text_survives_the_round_trip(self) -> None:
        # 2.31% of characters are accented; escaping them would be a silent corpus change.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = _system(directory, "s1_p1-s1", [_box("säu", 10)])
            crops = crop_syllables(record, directory / "out")
            manifest = directory / "train.jsonl"

            write_manifest(crops, manifest)

            self.assertIn("säu", manifest.read_text(encoding="utf-8"))


class TestDescribe(unittest.TestCase):
    def test_it_reports_how_much_of_valid_is_unseen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            train = crop_syllables(_system(directory, "a_p1-s1", [_box("known", 10)]),
                                   directory / "t")
            valid = crop_syllables(_system(directory, "b_p1-s1", [_box("unseen", 10)]),
                                   directory / "v")

            report = describe(train, valid)

        self.assertIn("100.0%", report)


if __name__ == "__main__":
    unittest.main()
