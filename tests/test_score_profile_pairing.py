import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.score_profile_pairing import (
    _find_score_musicxml,
    _profile_for_score,
    parse_ossq_stem,
    profile_and_part_for_sample,
)

_TWO_PART_SCORE = """<score-partwise>
  <part-list>
    <score-part id="P1">
      <part-name>Violin</part-name>
      <score-instrument id="P1-I1"><instrument-sound>strings.violin</instrument-sound></score-instrument>
    </score-part>
    <score-part id="P2">
      <part-name>Cello</part-name>
      <score-instrument id="P2-I1"><instrument-sound>strings.cello</instrument-sound></score-instrument>
    </score-part>
  </part-list>
  <part id="P1"><measure number="1"><attributes><clef><sign>G</sign><line>2</line></clef></attributes></measure></part>
  <part id="P2"><measure number="1"><attributes><clef><sign>F</sign><line>4</line></clef></attributes></measure></part>
</score-partwise>"""


class TestParseOssqStem(unittest.TestCase):
    def test_a_matching_stem_gives_score_id_and_one_based_part(self) -> None:
        self.assertEqual(parse_ossq_stem("sq7313978_0001_0001_2"), ("sq7313978", 2))

    def test_a_score_id_is_not_assumed_to_be_underscore_free(self) -> None:
        # rsplit-from-the-right semantics: only the last three underscore-separated
        # fields are page/system/part, whatever the score id itself contains.
        self.assertEqual(parse_ossq_stem("some_weird_id_0001_0001_1"), ("some_weird_id", 1))

    def test_a_non_matching_stem_returns_none(self) -> None:
        self.assertIsNone(parse_ossq_stem("not-an-ossq-stem"))

    def test_too_few_fields_returns_none(self) -> None:
        self.assertIsNone(parse_ossq_stem("sq7313978_0001"))


class TestFindAndProfileForScore(unittest.TestCase):
    def _make_corpus(self, tmp: str) -> None:
        work = Path(tmp) / "scores" / "Some,_Composer" / "Some_Piece"
        work.mkdir(parents=True)
        (work / "sq123.musicxml").write_text(_TWO_PART_SCORE, encoding="utf-8")

    def test_finds_the_whole_score_file_by_score_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()

            found = _find_score_musicxml(tmp, "sq123")

            self.assertIsNotNone(found)
            assert found is not None
            self.assertEqual(found.name, "sq123.musicxml")

    def test_a_missing_score_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()

            self.assertIsNone(_find_score_musicxml(tmp, "nonexistent"))

    def test_profile_for_score_extracts_both_parts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()
            _profile_for_score.cache_clear()

            profile = _profile_for_score(tmp, "sq123")

            self.assertIsNotNone(profile)
            assert profile is not None
            self.assertEqual(len(profile.parts), 2)
            self.assertEqual(profile.parts[0].instrument_family, "strings.violin")
            self.assertEqual(profile.parts[1].instrument_family, "strings.cello")


class TestProfileAndPartForSample(unittest.TestCase):
    def _make_corpus(self, tmp: str) -> None:
        work = Path(tmp) / "scores" / "Some,_Composer" / "Some_Piece"
        work.mkdir(parents=True)
        (work / "sq123.musicxml").write_text(_TWO_PART_SCORE, encoding="utf-8")

    def test_resolves_the_correct_part_by_one_based_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()
            _profile_for_score.cache_clear()

            result = profile_and_part_for_sample(tmp, "sq123_0001_0001_2")

            self.assertIsNotNone(result)
            assert result is not None
            profile, part = result
            self.assertEqual(len(profile.parts), 2)
            self.assertEqual(part.instrument_family, "strings.cello")

    def test_the_first_part_is_index_one_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()
            _profile_for_score.cache_clear()

            result = profile_and_part_for_sample(tmp, "sq123_0001_0001_1")

            assert result is not None
            self.assertEqual(result[1].instrument_family, "strings.violin")

    def test_an_out_of_range_part_index_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()
            _profile_for_score.cache_clear()

            self.assertIsNone(profile_and_part_for_sample(tmp, "sq123_0001_0001_9"))

    def test_a_non_ossq_stem_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(profile_and_part_for_sample(tmp, "some_other_corpus_sample"))

    def test_an_unresolvable_score_id_resolves_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._make_corpus(tmp)
            _find_score_musicxml.cache_clear()
            _profile_for_score.cache_clear()

            self.assertIsNone(profile_and_part_for_sample(tmp, "nonexistent_0001_0001_1"))


if __name__ == "__main__":
    unittest.main()
