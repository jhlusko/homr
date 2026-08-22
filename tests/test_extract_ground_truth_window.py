import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.ossq_ground_truth import extract_ground_truth_window

_MULTI_MOVEMENT_SCORE = """<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Violin</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><clef><sign>G</sign><line>2</line></clef>
      <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="3">
      <note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="1">
      <attributes><clef><sign>F</sign><line>4</line></clef>
      <time><beats>3</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>F</step><octave>3</octave></pitch><duration>3</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>3</octave></pitch><duration>3</duration></note>
    </measure>
  </part>
</score-partwise>"""


class TestExtractGroundTruthWindow(unittest.TestCase):
    def _write_score(self, tmp: str) -> Path:
        path = Path(tmp) / "sq1.musicxml"
        path.write_text(_MULTI_MOVEMENT_SCORE, encoding="utf-8")
        return path

    def test_extracts_only_the_requested_movements_measures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "out.musicxml"

            ok = extract_ground_truth_window(gt_path, 0, 2, 3, out_path)

            self.assertTrue(ok)
            tree = ET.parse(out_path)
            measures = tree.getroot().find(".//part").findall("measure")
            self.assertEqual([m.get("number") for m in measures], ["2", "3"])

    def test_does_not_splice_the_other_movements_copy_of_the_same_number(self) -> None:
        # Movement 1 also has a measure numbered "2" - a naive whole-file match would
        # keep both; this must keep only movement 0's copy.
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "out.musicxml"

            extract_ground_truth_window(gt_path, 0, 2, 2, out_path)

            tree = ET.parse(out_path)
            measures = tree.getroot().find(".//part").findall("measure")
            self.assertEqual(len(measures), 1)
            note = measures[0].find("note/pitch/step")
            self.assertEqual(note.text, "D")  # movement 0's measure 2, not movement 1's

    def test_carries_forward_clef_and_time_from_the_movements_own_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "out.musicxml"

            extract_ground_truth_window(gt_path, 0, 2, 3, out_path)

            tree = ET.parse(out_path)
            first_measure = tree.getroot().find(".//part").find("measure")
            clef = first_measure.find("attributes/clef/sign")
            self.assertIsNotNone(clef)
            self.assertEqual(clef.text, "G")  # movement 0's clef, not movement 1's F clef

    def test_the_second_movement_carries_forward_its_own_attributes_not_the_firsts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "out.musicxml"

            extract_ground_truth_window(gt_path, 1, 2, 2, out_path)

            tree = ET.parse(out_path)
            first_measure = tree.getroot().find(".//part").find("measure")
            self.assertEqual(first_measure.get("number"), "2")
            clef = first_measure.find("attributes/clef/sign")
            self.assertIsNotNone(clef)
            self.assertEqual(clef.text, "F")  # movement 1's own clef

    def test_merges_carried_attributes_with_a_partial_redeclaration(self) -> None:
        # Real corpus bug found via phase23: a window's first kept measure can already
        # redeclare *one* attribute (here, a time change) without redeclaring divisions,
        # since MusicXML only restates a child when it changes. The old "skip carrying
        # forward if <attributes> exists at all" logic left divisions at its implicit
        # default of 1 whenever this happened - this must merge in only the missing
        # child (divisions), not skip carrying forward entirely, and must not duplicate
        # or override the time change that's already there.
        score = """<score-partwise>
  <part-list><score-part id="P1"><part-name>Violin</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>24</divisions>
      <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>24</duration></note>
    </measure>
    <measure number="2">
      <attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>D</step><octave>5</octave></pitch><duration>24</duration></note>
    </measure>
  </part>
</score-partwise>"""
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = Path(tmp) / "sq1.musicxml"
            gt_path.write_text(score, encoding="utf-8")
            out_path = Path(tmp) / "out.musicxml"

            extract_ground_truth_window(gt_path, 0, 2, 2, out_path)

            tree = ET.parse(out_path)
            first_measure = tree.getroot().find(".//part").find("measure")
            attrs = first_measure.find("attributes")
            self.assertEqual(attrs.find("divisions").text, "24")
            self.assertEqual(attrs.find("time/beats").text, "3")  # not overridden

    def test_no_measure_in_range_returns_false_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "out.musicxml"

            ok = extract_ground_truth_window(gt_path, 0, 99, 100, out_path)

            self.assertFalse(ok)
            self.assertFalse(out_path.exists())

    def test_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gt_path = self._write_score(tmp)
            out_path = Path(tmp) / "nested" / "dir" / "out.musicxml"

            ok = extract_ground_truth_window(gt_path, 0, 2, 3, out_path)

            self.assertTrue(ok)
            self.assertTrue(out_path.exists())


if __name__ == "__main__":
    unittest.main()
