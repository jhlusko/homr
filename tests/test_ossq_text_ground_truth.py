import tempfile
import unittest
from pathlib import Path

from PIL import Image

from training.ocr.detector_patches import Sample
from training.ocr.detector_split import score_of, score_of_mask
from training.omr_datasets.ossq_text_ground_truth import (
    matches_for_score,
    pages_of,
    score_dirs,
    score_id_of,
)

MSCX = (
    b"<museScore><Score><Staff><Measure>"
    b"<Dynamic><subtype>ff</subtype></Dynamic>"
    b"<Tempo><text>Allegro</text></Tempo>"
    b"<StaffText><style>Expression</style><text>ben marcato</text></StaffText>"
    b"</Measure></Staff></Score></museScore>"
)


def _page_file(path: Path) -> None:
    """A real, decodable PNG - `ocr_page` decodes the file itself now, so a fixture
    holding placeholder bytes would silently produce no lines and no matches."""
    Image.new("RGB", (40, 30), "white").save(path)


def _work(root: Path, composer: str, work: str, score: str, pages: int = 2) -> Path:
    d = root / composer / work
    (d / "images" / "scanned" / "original").mkdir(parents=True)
    (d / f"{score}.mscx").write_bytes(MSCX)
    for i in range(pages):
        _page_file(d / "images" / "scanned" / "original" / f"{score}:{i:04d}.png")
    return d


class _Reader:
    """Stands in for RapidOCR: returns fixed boxes regardless of the image."""

    def __init__(self, texts):
        self.texts = texts

    def __call__(self, path):
        class R:
            pass

        r = R()
        r.txts = list(self.texts)
        r.scores = [0.9] * len(self.texts)
        r.boxes = [
            [[i * 10, 0], [i * 10 + 8, 0], [i * 10 + 8, 6], [i * 10, 6]]
            for i in range(len(self.texts))
        ]
        return r


class TestDiscovery(unittest.TestCase):
    def test_it_finds_works_with_both_a_source_and_scans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _work(root, "Beethoven", "Op.133", "sq1")

            self.assertEqual(len(score_dirs(root)), 1)

    def test_a_work_without_scans_is_skipped(self) -> None:
        # Skipping here rather than failing later on a missing directory.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "Beethoven" / "Op.18"
            d.mkdir(parents=True)
            (d / "sq2.mscx").write_bytes(MSCX)

            self.assertEqual(score_dirs(root), [])

    def test_the_score_id_comes_from_the_mscx_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq10502527")

            self.assertEqual(score_id_of(d), "sq10502527")

    def test_pages_are_returned_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq1", pages=3)

            self.assertEqual([p.name for p in pages_of(d)], ["sq1:0000.png", "sq1:0001.png",
                                                             "sq1:0002.png"])


class TestMatchesForScore(unittest.TestCase):
    def test_it_confirms_marks_of_every_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq1", pages=1)

            matches = matches_for_score(_Reader(["ff", "Allegro", "ben marcato"]), d)

        self.assertEqual(
            sorted(m["kind"] for m in matches), ["dynamic", "expression", "tempo"]
        )

    def test_ocr_text_that_matches_nothing_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq1", pages=1)

            matches = matches_for_score(_Reader(["Violoncello", "42"]), d)

        self.assertEqual(matches, [])

    def test_each_box_is_claimed_by_only_one_kind(self) -> None:
        # "cresc." is written as both an expression and a staff text in this corpus;
        # emitting one box twice under two classes would leave the mask holding
        # whichever happened to rasterise last.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "X" / "Y"
            (d / "images" / "scanned" / "original").mkdir(parents=True)
            (d / "sq9.mscx").write_bytes(
                b"<museScore><Measure>"
                b"<StaffText><text>cresc.</text></StaffText>"
                b"<StaffText><style>Expression</style><text>cresc.</text></StaffText>"
                b"</Measure></museScore>"
            )
            _page_file(d / "images" / "scanned" / "original" / "sq9:0001.png")

            matches = matches_for_score(_Reader(["cresc."]), d)

        self.assertEqual(len(matches), 1)

    def test_matches_record_an_absolute_page_path(self) -> None:
        # OSSQ pages are not laid out as <pngs>/<score>/<page>, so the mask builder
        # is given the path directly rather than a name to join.
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq1", pages=1)

            matches = matches_for_score(_Reader(["ff"]), d)

            self.assertTrue(matches[0]["page_image"].startswith("/"))
            self.assertTrue(Path(matches[0]["page_image"]).is_file())

    def test_every_page_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = _work(Path(tmp), "Beethoven", "Op.133", "sq1", pages=3)

            matches = matches_for_score(_Reader(["ff"]), d)

        self.assertEqual(len({m["page_image"] for m in matches}), 3)


class TestScoreOfMask(unittest.TestCase):
    def test_ossq_pages_would_all_share_one_folder_name(self) -> None:
        # The trap this exists for: every OSSQ page sits in a folder called
        # "original", so the folder rule reports one score for the whole corpus and
        # the split silently stops being score-disjoint while still claiming to be.
        sample = Sample(
            "/data/ossq/Beethoven/Op.133/images/scanned/original/sq1:0001.png",
            "/data/masks/sq1_sq1:0001.mask.png",
        )

        self.assertEqual(score_of(sample), "original")
        self.assertEqual(score_of_mask(sample), "sq1")

    def test_the_mask_rule_still_works_for_the_other_corpora(self) -> None:
        synthetic = Sample("/d/mbox/4919798_p1-s3/a.png", "/d/masks/4919798_p1-s3-1.mask.png")
        lieder = Sample("/d/imslp_pngs/IMSLP10416/p3.png", "/d/masks/IMSLP10416_p3.mask.png")

        self.assertEqual(score_of_mask(synthetic), "4919798")
        self.assertEqual(score_of_mask(lieder), "IMSLP10416")


class TestOcrPageDecoding(unittest.TestCase):
    """`ocr_page` must decode the image itself.

    RapidOCR's own file loader returns zero boxes, silently, on palette-mode PNGs -
    which is every OSSQ scan and no Lieder scan. Passing a path therefore produced an
    empty dataset for one whole corpus while looking like a successful run.
    """

    def _page(self, directory: Path, mode: str) -> Path:
        from PIL import Image

        path = directory / f"page_{mode}.png"
        image = Image.new("RGB", (40, 30), "white")
        image.convert(mode).save(path)
        return path

    def test_a_palette_png_is_decoded_and_passed_as_an_array(self) -> None:
        import numpy as np

        from training.omr_datasets.ocr_first_text_ground_truth import ocr_page

        seen = {}

        class _Reader:
            def __call__(self, image):
                seen["type"] = type(image)
                r = type("R", (), {})()
                r.boxes = None
                r.txts = None
                r.scores = None
                return r

        with tempfile.TemporaryDirectory() as tmp:
            ocr_page(_Reader(), self._page(Path(tmp), "P"))

        self.assertIs(seen["type"], np.ndarray)

    def test_an_unreadable_file_gives_no_lines_rather_than_raising(self) -> None:
        from training.omr_datasets.ocr_first_text_ground_truth import ocr_page

        class _Reader:
            def __call__(self, image):  # pragma: no cover - must not be reached
                raise AssertionError("should not be called for an unreadable file")

        self.assertEqual(ocr_page(_Reader(), Path("/nonexistent/page.png")), [])


if __name__ == "__main__":
    unittest.main()
