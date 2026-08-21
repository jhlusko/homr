import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import (
    measure_start_for_system,
    piece_dir,
    real_ground_truth_path,
    score_and_page,
)


def _make_piece(root: Path) -> Path:
    """A minimal piece directory: real ground truth at the top level, one page image
    under images/scanned/original/, and one system's metadata under
    metadata/scanned/systemwise/ - mirroring the real ossq-omr layout this module reads."""
    piece = root / "scores" / "Some Composer" / "Some Piece"
    images_dir = piece / "images" / "scanned" / "original"
    images_dir.mkdir(parents=True)
    (images_dir / "sq1234:0005.png").write_bytes(b"")
    (piece / "sq1234.musicxml").write_text("<score-partwise/>", encoding="utf-8")

    systemwise = piece / "metadata" / "scanned" / "systemwise"
    systemwise.mkdir(parents=True)
    (systemwise / "sq1234:0005:0001.yaml").write_text(
        "score_id: sq1234\npage_idx: 4\nsystem_idx: 1\nmeasure_start: 40\nmeasure_end: 43\n",
        encoding="utf-8",
    )
    return images_dir / "sq1234:0005.png"


class TestScoreAndPage(unittest.TestCase):
    def test_splits_score_id_and_page(self) -> None:
        self.assertEqual(score_and_page(Path("sq10675759:0024.png")), ("sq10675759", "0024"))


class TestPieceDir(unittest.TestCase):
    def test_resolves_up_from_images_scanned_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            self.assertEqual(piece_dir(image).name, "Some Piece")


class TestRealGroundTruthPath(unittest.TestCase):
    def test_finds_the_top_level_whole_score_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            gt = real_ground_truth_path(image)
            self.assertIsNotNone(gt)
            self.assertEqual(gt.name, "sq1234.musicxml")

    def test_missing_ground_truth_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            (image.parents[3] / "sq1234.musicxml").unlink()
            self.assertIsNone(real_ground_truth_path(image))


class TestMeasureStartForSystem(unittest.TestCase):
    def test_reads_measure_start_from_scanned_systemwise_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            self.assertEqual(measure_start_for_system(image, system_index=0), 40)

    def test_no_metadata_for_the_system_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            self.assertIsNone(measure_start_for_system(image, system_index=1))

    def test_falls_back_to_unaligned_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            piece = image.parents[3]
            unaligned = piece / "metadata" / "unaligned"
            unaligned.mkdir(parents=True)
            (unaligned / "sq1234:0005:0002.yaml").write_text(
                "score_id: sq1234\npage_idx: 4\nsystem_idx: 2\nmeasure_start: 44\nmeasure_end: 46\n",
                encoding="utf-8",
            )
            self.assertEqual(measure_start_for_system(image, system_index=1), 44)

    def test_a_non_numeric_placeholder_is_treated_as_no_mapping(self) -> None:
        # Real corpus metadata has been observed carrying "X2" here - presumably
        # marking an alignment the corpus itself isn't sure of.
        with tempfile.TemporaryDirectory() as tmp:
            image = _make_piece(Path(tmp))
            systemwise = image.parents[3] / "metadata" / "scanned" / "systemwise"
            (systemwise / "sq1234:0005:0002.yaml").write_text(
                "score_id: sq1234\npage_idx: 4\nsystem_idx: 2\nmeasure_start: X2\nmeasure_end: X4\n",
                encoding="utf-8",
            )
            self.assertIsNone(measure_start_for_system(image, system_index=1))


if __name__ == "__main__":
    unittest.main()
