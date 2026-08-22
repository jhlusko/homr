import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import _systemwise_entries_cached
from training.omr_datasets.split_ground_truth_by_system import fragment_path, split_piece

_SCORE = """<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Violin</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="3">
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="4">
      <note><pitch><step>F</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>"""


def _make_piece(root: Path) -> tuple[Path, Path]:
    piece = root / "scores" / "Some Composer" / "Some Piece"
    piece.mkdir(parents=True)
    gt_path = piece / "sq1.musicxml"
    gt_path.write_text(_SCORE, encoding="utf-8")

    systemwise = piece / "metadata" / "scanned" / "systemwise"
    systemwise.mkdir(parents=True)
    (systemwise / "sq1:0001:0001.yaml").write_text(
        "score_id: sq1\npage_idx: 0\nsystem_idx: 1\nmeasure_start: 1\nmeasure_end: 2\n",
        encoding="utf-8",
    )
    (systemwise / "sq1:0001:0002.yaml").write_text(
        "score_id: sq1\npage_idx: 0\nsystem_idx: 2\nmeasure_start: 3\nmeasure_end: 4\n",
        encoding="utf-8",
    )
    return gt_path, piece


class TestSplitPiece(unittest.TestCase):
    def setUp(self) -> None:
        _systemwise_entries_cached.cache_clear()

    def test_writes_one_fragment_per_system(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, piece_dir = _make_piece(Path(tmp))

            written, skipped = split_piece(gt_path, piece_dir, "sq1")

            self.assertEqual(written, 2)
            self.assertEqual(skipped, 0)

    def test_each_fragment_has_the_expected_measures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, piece_dir = _make_piece(Path(tmp))

            split_piece(gt_path, piece_dir, "sq1")

            frag1 = fragment_path(piece_dir, 1, 1)
            frag2 = fragment_path(piece_dir, 1, 2)
            self.assertTrue(frag1.exists())
            self.assertTrue(frag2.exists())

            measures1 = ET.parse(frag1).getroot().find(".//part").findall("measure")
            measures2 = ET.parse(frag2).getroot().find(".//part").findall("measure")
            self.assertEqual([m.get("number") for m in measures1], ["1", "2"])
            self.assertEqual([m.get("number") for m in measures2], ["3", "4"])

    def test_a_fragment_carries_forward_the_opening_time_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path, piece_dir = _make_piece(Path(tmp))

            split_piece(gt_path, piece_dir, "sq1")

            frag2 = fragment_path(piece_dir, 1, 2)
            first_measure = ET.parse(frag2).getroot().find(".//part").find("measure")
            time_el = first_measure.find("attributes/time/beats")
            self.assertIsNotNone(time_el)
            self.assertEqual(time_el.text, "4")

    def test_a_piece_with_no_aligned_metadata_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            piece = Path(tmp) / "scores" / "Some Composer" / "Some Piece"
            piece.mkdir(parents=True)
            gt_path = piece / "sq2.musicxml"
            gt_path.write_text(_SCORE, encoding="utf-8")

            written, skipped = split_piece(gt_path, piece, "sq2")

            self.assertEqual(written, 0)
            self.assertEqual(skipped, 0)


if __name__ == "__main__":
    unittest.main()
