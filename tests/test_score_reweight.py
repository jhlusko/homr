import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.score_reweight import (
    ImageWeight,
    measure_contrast,
    repeat_counts,
    reweight_index,
)


def _image(directory: Path, name: str, level: int) -> str:
    """A page at a given gray level - lower `level` variation means lower contrast."""
    path = directory / f"{name}.png"
    page = np.full((40, 40), 200, dtype=np.uint8)
    page[10:30, 10:30] = level
    cv2.imwrite(str(path), page)
    return str(path)


class TestMeasureContrast(unittest.TestCase):
    def test_a_faint_image_measures_lower_than_a_crisp_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            faint = _image(directory, "faint", 180)
            crisp = _image(directory, "crisp", 0)

            values = measure_contrast([faint, crisp])

        self.assertLess(values[faint], values[crisp])

    def test_an_unreadable_path_is_skipped_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crisp = _image(Path(tmp), "crisp", 0)

            values = measure_contrast([crisp, str(Path(tmp) / "absent.png")])

        self.assertEqual(list(values), [crisp])


class TestRepeatCounts(unittest.TestCase):
    def test_faintest_images_get_the_most_repeats(self) -> None:
        weights = repeat_counts({"a": 50, "b": 100, "c": 250}, floor_percentile=50)

        self.assertGreater(weights["a"].repeats, weights["c"].repeats)

    def test_the_floor_is_a_percentile_not_a_fixed_value(self) -> None:
        # A percentile stays stable across corpora of different average quality; a fixed
        # contrast threshold would need re-tuning for every new source.
        low_quality = repeat_counts({"a": 50, "b": 60, "c": 70, "d": 80}, floor_percentile=25)
        high_quality = repeat_counts(
            {"a": 200, "b": 210, "c": 220, "d": 230}, floor_percentile=25
        )

        self.assertGreater(low_quality["a"].repeats, 1)
        self.assertGreater(high_quality["a"].repeats, 1)

    def test_images_at_or_above_the_floor_are_untouched(self) -> None:
        weights = repeat_counts({"a": 50, "b": 100, "c": 250, "d": 260}, floor_percentile=25)

        self.assertEqual(weights["d"].repeats, 1)

    def test_the_repeat_count_is_capped(self) -> None:
        weights = repeat_counts({"a": 1, "b": 250}, floor_percentile=90, max_repeats=4)

        self.assertLessEqual(weights["a"].repeats, 4)

    def test_an_empty_input_produces_no_weights(self) -> None:
        self.assertEqual(repeat_counts({}), {})

    def test_uniform_contrast_leaves_everything_at_one(self) -> None:
        # Floor equals the worst value, so there is no faint tail to boost.
        weights = repeat_counts({"a": 200, "b": 200, "c": 200}, floor_percentile=25)

        self.assertTrue(all(w.repeats == 1 for w in weights.values()))


class TestReweightIndex(unittest.TestCase):
    def test_lines_are_repeated_by_their_images_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                "/data/a.png,/data/a.txt\n/data/b.png,/data/b.txt\n", encoding="utf-8"
            )
            weights = {
                "/data/a.png": ImageWeight("/data/a.png", 50.0, 3),
                "/data/b.png": ImageWeight("/data/b.png", 250.0, 1),
            }

            before, after = reweight_index(index, weights, directory / "out.txt")
            lines = (directory / "out.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(before, 2)
        self.assertEqual(after, 4)
        self.assertEqual(sum(1 for line in lines if "a.png" in line), 3)
        self.assertEqual(sum(1 for line in lines if "b.png" in line), 1)

    def test_an_unmeasured_image_is_kept_once_not_guessed_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text("/data/unmeasured.png,/data/x.txt\n", encoding="utf-8")

            _, after = reweight_index(index, {}, directory / "out.txt")

        self.assertEqual(after, 1)

    def test_blank_lines_are_dropped_not_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text("/data/a.png,/data/a.txt\n\n\n", encoding="utf-8")
            weights = {"/data/a.png": ImageWeight("/data/a.png", 50.0, 4)}

            _, after = reweight_index(index, weights, directory / "out.txt")

        self.assertEqual(after, 4)

    def test_it_reweights_by_image_path_not_by_score_name(self) -> None:
        # The design this replaces keyed on score identity from the token filename, which
        # could not work: OSSQ's split is by score, so scores measured as faint in
        # validation never appear in the training index at all. Weighting by the image path
        # itself has no such dependency.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                "/data/sq111_p1-s1.png,/data/sq111_p1-s1.txt\n"
                "/data/sq222_p1-s1.png,/data/sq222_p1-s1.txt\n",
                encoding="utf-8",
            )
            weights = {"/data/sq111_p1-s1.png": ImageWeight("x", 50.0, 3)}

            _, after = reweight_index(index, weights, directory / "out.txt")
            lines = (directory / "out.txt").read_text(encoding="utf-8").splitlines()

        self.assertEqual(sum(1 for line in lines if "sq111" in line), 3)
        self.assertEqual(sum(1 for line in lines if "sq222" in line), 1)


if __name__ == "__main__":
    unittest.main()
