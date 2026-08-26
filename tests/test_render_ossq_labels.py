import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.render_ossq_labels import (
    MARGIN,
    pending,
    token_paths,
    trim,
    trim_in_place,
)


def _page(with_ink: bool = True) -> np.ndarray:
    page = np.full((400, 600, 3), 255, dtype=np.uint8)
    if with_ink:
        page[180:220, 100:300] = 0
    return page


class TestTrim(unittest.TestCase):
    def test_it_cuts_the_page_down_to_the_inked_region(self) -> None:
        # MuseScore renders a full A4 sheet for one staff. Beside a tightly cropped
        # photograph, the untrimmed notes are a fraction of the size of the ones they
        # are compared against, which defeats showing pictures at all.
        trimmed = trim(_page())

        self.assertLess(trimmed.shape[0], 400)
        self.assertLess(trimmed.shape[1], 600)

    def test_the_ink_survives_with_a_margin(self) -> None:
        trimmed = trim(_page())

        self.assertEqual(trimmed.shape[0], 40 + 2 * MARGIN)
        self.assertEqual(trimmed.shape[1], 200 + 2 * MARGIN)

    def test_an_all_white_page_is_returned_unchanged(self) -> None:
        # Collapsing to zero size would produce an unreadable file rather than an
        # obviously empty one.
        page = _page(with_ink=False)

        self.assertEqual(trim(page).shape, page.shape)

    def test_a_margin_never_runs_off_the_page(self) -> None:
        page = np.full((30, 30, 3), 255, dtype=np.uint8)
        page[0, 0] = 0
        page[29, 29] = 0

        self.assertEqual(trim(page).shape[:2], (30, 30))

    def test_light_grey_counts_as_paper(self) -> None:
        page = np.full((100, 100, 3), 250, dtype=np.uint8)
        page[40:50, 40:50] = 10

        self.assertEqual(trim(page).shape[0], 10 + 2 * MARGIN)


class TestTrimInPlace(unittest.TestCase):
    def test_it_rewrites_the_file_smaller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.png"
            cv2.imwrite(str(path), _page())

            self.assertTrue(trim_in_place(path))
            self.assertLess(cv2.imread(str(path)).shape[0], 400)

    def test_an_unreadable_file_is_reported_rather_than_raising(self) -> None:
        self.assertFalse(trim_in_place(Path("/nonexistent/r.png")))


class TestQueue(unittest.TestCase):
    def test_token_paths_reads_the_second_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "i.txt"
            index.write_text("/a/x.png,/a/x.txt\n/a/y.png,/a/y.txt\n", encoding="utf-8")

            self.assertEqual([p.name for p in token_paths(index)], ["x.txt", "y.txt"])

    def test_pending_skips_what_is_already_rendered(self) -> None:
        # Re-running must resume rather than repeat: rendering is the slow part.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cv2.imwrite(str(out / "x.png"), _page())

            todo = pending([Path("/a/x.txt"), Path("/a/y.txt")], out)

        self.assertEqual([p.name for p in todo], ["y.txt"])


if __name__ == "__main__":
    unittest.main()
