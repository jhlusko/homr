import tempfile
import unittest
import zipfile
from pathlib import Path

from training.omr_datasets.lyric_survey import (
    collect,
    heaps_exponent,
    out_of_vocabulary,
    vocabulary_growth,
)

SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    {notes}
  </measure></part>
</score-partwise>
"""


def _note(step: str, lyrics: str = "") -> str:
    return (
        f"<note><pitch><step>{step}</step><octave>4</octave></pitch>"
        f"<duration>1</duration><type>quarter</type>{lyrics}</note>"
    )


def _lyric(text: str, syllabic: str = "single", number: str = "1") -> str:
    return f'<lyric number="{number}"><syllabic>{syllabic}</syllabic><text>{text}</text></lyric>'


def _score(directory: Path, name: str, notes: str) -> Path:
    path = directory / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("META-INF/container.xml", "<container/>")
        archive.writestr("score.xml", SCORE.format(notes=notes))
    return path


class TestCollect(unittest.TestCase):
    def test_syllables_and_characters_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _score(Path(tmp), "a.mxl", _note("C", _lyric("Ê")) + _note("D", _lyric("ter")))

            survey = collect([path])

        self.assertEqual(survey.occurrences, 2)
        self.assertIn("Ê", survey.characters)

    def test_a_melisma_is_counted_as_one_syllable_over_several_notes(self) -> None:
        # 1:1 note-to-syllable is what a per-token head would assume, and this is where
        # that assumption breaks.
        notes = _note("C", _lyric("A", "begin")) + _note("D") + _note("E", _lyric("men", "end"))
        with tempfile.TemporaryDirectory() as tmp:
            survey = collect([_score(Path(tmp), "a.mxl", notes)])

        self.assertEqual(survey.lyric_notes, 2)
        self.assertEqual(survey.vocal_notes, 3)

    def test_several_verses_on_one_note_are_all_kept(self) -> None:
        notes = _note("C", _lyric("first") + _lyric("second", number="2"))
        with tempfile.TemporaryDirectory() as tmp:
            survey = collect([_score(Path(tmp), "a.mxl", notes)])

        self.assertEqual(survey.verses_per_note[2], 1)
        self.assertEqual(survey.highest_verse, 2)

    def test_rests_and_chord_members_are_not_counted_as_notes(self) -> None:
        # A chord member repeats a notehead already counted; a rest cannot bear a lyric.
        notes = (
            _note("C", _lyric("la"))
            + "<note><chord/><pitch><step>E</step><octave>4</octave></pitch>"
            "<duration>1</duration><type>quarter</type></note>"
            + "<note><rest/><duration>1</duration><type>quarter</type></note>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            survey = collect([_score(Path(tmp), "a.mxl", notes)])

        self.assertEqual(survey.vocal_notes, 1)

    def test_parts_without_lyrics_are_left_out_of_the_note_count(self) -> None:
        # The denominator is notes in lyric-carrying parts; counting the piano's notes
        # would make the lyric-bearing share look far smaller than it is.
        with tempfile.TemporaryDirectory() as tmp:
            survey = collect([_score(Path(tmp), "a.mxl", _note("C"))])

        self.assertEqual(survey.vocal_notes, 0)


class TestOutOfVocabulary(unittest.TestCase):
    def test_syllables_absent_from_training_are_reported_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [
                _score(directory, f"{index}.mxl", _note("C", _lyric("known")))
                for index in range(4)
            ]
            paths.append(_score(directory, "9.mxl", _note("C", _lyric("unseen"))))

            mass, types, size = out_of_vocabulary(paths, holdout=0.2)

        self.assertEqual(mass, 1.0)
        self.assertEqual(types, 1.0)
        self.assertEqual(size, 1)

    def test_a_shared_vocabulary_reports_nothing_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [
                _score(directory, f"{index}.mxl", _note("C", _lyric("same")))
                for index in range(5)
            ]

            mass, types, _ = out_of_vocabulary(paths, holdout=0.2)

        self.assertEqual((mass, types), (0.0, 0.0))


class TestVocabularyGrowth(unittest.TestCase):
    """A vocabulary count from one corpus is bounded by that corpus. What says whether a
    closed set is possible is whether the count is still climbing when the scores run out."""

    def test_growth_is_reported_at_each_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [
                _score(directory, f"{index}.mxl", _note("C", _lyric(f"syl{index}")))
                for index in range(8)
            ]

            growth = vocabulary_growth(paths, steps=4)

        self.assertEqual([scores for scores, _, _ in growth], [2, 4, 6, 8])
        self.assertEqual([types for _, _, types in growth], [2, 4, 6, 8])

    def test_a_corpus_that_never_repeats_itself_grows_linearly(self) -> None:
        # beta near 1 means every new score brings new words - the case where a closed
        # vocabulary cannot work at any size.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [
                _score(directory, f"{index}.mxl", _note("C", _lyric(f"syl{index}")))
                for index in range(8)
            ]

            self.assertAlmostEqual(heaps_exponent(vocabulary_growth(paths, steps=4)), 1.0, places=2)

    def test_a_corpus_that_only_repeats_itself_does_not_grow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [
                _score(directory, f"{index}.mxl", _note("C", _lyric("same")))
                for index in range(8)
            ]

            self.assertAlmostEqual(heaps_exponent(vocabulary_growth(paths, steps=4)), 0.0, places=2)

    def test_too_few_points_to_fit_is_zero_not_an_error(self) -> None:
        self.assertEqual(heaps_exponent([]), 0.0)


if __name__ == "__main__":
    unittest.main()
