import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.cross_staff_coherence import system_measure_curve
from training.omr_datasets.ossq_ground_truth import fragment_path
from training.omr_datasets.score_profile_pairing import _find_score_musicxml

_TWO_MATCHING_PARTS = """<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Violin I</part-name></score-part>
    <score-part id="P2"><part-name>Violin II</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>G</step><octave>3</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2">
      <note><pitch><step>A</step><octave>3</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
</score-partwise>"""

# Three parts, the third one's second measure genuinely wrong (3 quarters, not 4) -
# a labeling defect like the ones ossq_measure_length_audit.py's corpus audit found,
# which is exactly why the median (2 of 3 agree on 4) is the right target, not any one
# part's own value.
_THREE_PARTS_ONE_DEFECTIVE = """<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Violin I</part-name></score-part>
    <score-part id="P2"><part-name>Violin II</part-name></score-part>
    <score-part id="P3"><part-name>Viola</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>G</step><octave>3</octave></pitch><duration>4</duration></note>
    </measure>
  </part>
  <part id="P3">
    <measure number="1">
      <attributes><divisions>1</divisions></attributes>
      <note><pitch><step>E</step><octave>3</octave></pitch><duration>3</duration></note>
    </measure>
  </part>
</score-partwise>"""


def _write_fragment(tmp: str, score_id: str, xml: str, page: int = 5, system_num: int = 1) -> None:
    piece = Path(tmp) / "scores" / "Some Composer" / "Some Piece"
    piece.mkdir(parents=True, exist_ok=True)
    (piece / f"{score_id}.musicxml").write_text(
        "<score-partwise><part-list/></score-partwise>", encoding="utf-8"
    )
    frag = fragment_path(piece, page, system_num)
    frag.parent.mkdir(parents=True, exist_ok=True)
    frag.write_text(xml, encoding="utf-8")


class TestSystemMeasureCurve(unittest.TestCase):
    def setUp(self) -> None:
        _find_score_musicxml.cache_clear()

    def test_cumulative_whole_note_curve_across_two_agreeing_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fragment(tmp, "sq123", _TWO_MATCHING_PARTS)

            curve = system_measure_curve(tmp, "sq123_0005_0001_1")

            # Both parts agree: measure 1 = 4 quarters (1 whole note), measure 2 = 4
            # quarters too (1 whole note) - cumulative 1.0, then 2.0.
            self.assertEqual(curve, [1.0, 2.0])

    def test_takes_the_median_not_a_single_defective_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fragment(tmp, "sq123", _THREE_PARTS_ONE_DEFECTIVE)

            curve = system_measure_curve(tmp, "sq123_0005_0001_1")

            # 2 of 3 parts say 4 quarters (1 whole note); the median ignores the
            # defective 3-quarter part rather than being dragged toward it.
            self.assertEqual(curve, [1.0])

    def test_a_non_ossq_stem_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(system_measure_curve(tmp, "some_other_corpus_sample"))

    def test_an_unresolvable_score_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fragment(tmp, "sq123", _TWO_MATCHING_PARTS)
            self.assertIsNone(system_measure_curve(tmp, "nonexistent_0005_0001_1"))

    def test_no_fragment_for_this_page_system_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _write_fragment(tmp, "sq123", _TWO_MATCHING_PARTS)
            # system_index 5 (stem system "0006") has no fragment written.
            self.assertIsNone(system_measure_curve(tmp, "sq123_0005_0006_1"))


if __name__ == "__main__":
    unittest.main()
