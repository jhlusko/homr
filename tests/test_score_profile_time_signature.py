import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.score_profile_pairing import _find_score_musicxml, _profile_for_score
from training.omr_datasets.score_profile_time_signature import (
    parse_ossq_stem_full,
    time_signature_for_sample,
)

_SCORE_WITH_A_TIME_CHANGE = """<score-partwise>
  <part-list>
    <score-part id="P1"><part-name>Violin</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
    </measure>
    <measure number="20">
      <attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
    </measure>
    <measure number="40"></measure>
  </part>
</score-partwise>"""


def _make_corpus(tmp: str, measure_start: int = 40) -> None:
    piece = Path(tmp) / "scores" / "Some Composer" / "Some Piece"
    piece.mkdir(parents=True)
    (piece / "sq123.musicxml").write_text(_SCORE_WITH_A_TIME_CHANGE, encoding="utf-8")

    systemwise = piece / "metadata" / "scanned" / "systemwise"
    systemwise.mkdir(parents=True)
    (systemwise / "sq123:0005:0001.yaml").write_text(
        f"score_id: sq123\npage_idx: 4\nsystem_idx: 1\n"
        f"measure_start: {measure_start}\nmeasure_end: {measure_start + 3}\n",
        encoding="utf-8",
    )


class TestParseOssqStemFull(unittest.TestCase):
    def test_a_matching_stem_gives_all_four_fields(self) -> None:
        self.assertEqual(
            parse_ossq_stem_full("sq7313978_0005_0001_2"),
            ("sq7313978", "0005", 0, 1),
        )

    def test_a_non_matching_stem_returns_none(self) -> None:
        self.assertIsNone(parse_ossq_stem_full("not-an-ossq-stem"))


class TestTimeSignatureForSample(unittest.TestCase):
    def setUp(self) -> None:
        _find_score_musicxml.cache_clear()
        _profile_for_score.cache_clear()

    def test_reads_the_time_signature_in_effect_at_the_targeted_measure(self) -> None:
        # measure_start=40: the target measure is "40", which does not itself declare
        # a <time> - the last one declared before it (measure "20"'s 3/4) applies.
        with tempfile.TemporaryDirectory() as tmp:
            _make_corpus(tmp, measure_start=40)

            result = time_signature_for_sample(tmp, "sq123_0005_0001_1")

            self.assertEqual(result, "3/4")

    def test_a_measure_before_any_change_gets_the_opening_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # A system starting at measure 1 itself - only the opening 4/4 applies.
            _make_corpus(tmp, measure_start=1)

            result = time_signature_for_sample(tmp, "sq123_0005_0001_1")

            self.assertEqual(result, "4/4")

    def test_a_non_ossq_stem_resolves_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(time_signature_for_sample(tmp, "some_other_corpus_sample"), "")

    def test_an_unresolvable_score_resolves_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_corpus(tmp)
            self.assertEqual(
                time_signature_for_sample(tmp, "nonexistent_0005_0001_1"), ""
            )

    def test_no_alignment_metadata_for_this_system_resolves_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_corpus(tmp)
            # system_index 5 (stem system "0006") has no systemwise metadata at all.
            self.assertEqual(
                time_signature_for_sample(tmp, "sq123_0005_0006_1"), ""
            )

    def test_an_out_of_range_part_index_resolves_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_corpus(tmp)
            self.assertEqual(
                time_signature_for_sample(tmp, "sq123_0005_0001_9"), ""
            )


if __name__ == "__main__":
    unittest.main()
