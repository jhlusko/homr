import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from training.omr_datasets.beam_materialization_check import (
    BeamVector,
    beam_vectors,
    compare,
)

Notes = Sequence[tuple[str | None, BeamVector]]

_SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part id="P1"><measure number="1">
  <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>
    <type>eighth</type>{first}</note>
  <note><chord/><pitch><step>E</step><octave>5</octave></pitch><duration>1</duration>
    <type>eighth</type></note>
  <note><rest/><duration>1</duration><type>eighth</type></note>
  <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration>
    <type>eighth</type>{second}</note>
</measure></part></score-partwise>
"""


def _vectors(first: str = "", second: str = "") -> Notes:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.musicxml"
        path.write_text(_SCORE.format(first=first, second=second), encoding="utf-8")
        return beam_vectors(path)


class TestBeamVectors(unittest.TestCase):
    def test_chord_members_and_rests_are_skipped(self) -> None:
        # Chord members carry no beams of their own and rests none at all; including
        # them would misalign the two sides of the comparison note by note.
        self.assertEqual(len(_vectors()), 2)

    def test_vector_captures_level_and_state(self) -> None:
        vectors = _vectors(first="<beam number='1'>begin</beam><beam number='2'>begin</beam>")

        self.assertEqual(vectors[0][1], (("1", "begin"), ("2", "begin")))

    def test_a_note_with_no_beams_has_an_empty_vector(self) -> None:
        self.assertEqual(_vectors()[0][1], ())


class TestCompare(unittest.TestCase):
    def test_identical_sides_are_all_unchanged(self) -> None:
        side: Notes = [("eighth", (("1", "begin"),)), ("eighth", (("1", "end"),))]

        result = compare(side, list(side))

        self.assertEqual(
            (result.unchanged, result.gained, result.lost, result.changed), (2, 0, 0, 0)
        )

    def test_gaining_a_beam_is_the_ambiguity_signal(self) -> None:
        before: Notes = [("eighth", ())]
        after: Notes = [("eighth", (("1", "begin"),))]
        result = compare(before, after)

        self.assertEqual(result.gained, 1)
        self.assertEqual(result.ambiguity_rate, 1.0)
        self.assertTrue(result.materialization_needed)

    def test_losing_a_beam_counts_as_instability_not_ambiguity(self) -> None:
        before: Notes = [("eighth", (("1", "end"),))]
        after: Notes = [("eighth", ())]
        result = compare(before, after)

        self.assertEqual((result.gained, result.lost), (0, 1))
        self.assertEqual(result.ambiguity_rate, 0.0)
        self.assertFalse(result.round_trip_safe)

    def test_a_rewritten_vector_counts_as_instability(self) -> None:
        before: Notes = [("eighth", (("1", "continue"),))]
        after: Notes = [("eighth", (("1", "end"),))]
        result = compare(before, after)

        self.assertEqual(result.changed, 1)
        self.assertFalse(result.round_trip_safe)

    def test_a_note_count_mismatch_is_reported_not_aligned(self) -> None:
        before: Notes = [("eighth", ())]
        after: Notes = [("eighth", ()), ("eighth", ())]
        result = compare(before, after)

        self.assertEqual(result.notes, 0)
        self.assertEqual(len(result.skipped_scores), 1)

    def test_one_stray_note_does_not_trigger_materialization(self) -> None:
        # The corpus result: 1 note in 172,607 gained a beam. That is noise, and a
        # verdict that fires on a raw count rather than a rate would call for a whole
        # pipeline stage on the strength of it.
        side: Notes = [("eighth", (("1", "begin"),))] * 999 + [("eighth", ())]
        after: Notes = [("eighth", (("1", "begin"),))] * 1000

        result = compare(side, after)

        self.assertEqual(result.gained, 1)
        self.assertFalse(result.materialization_needed)

    def test_verdict_when_nothing_is_ambiguous_but_the_round_trip_rewrites(self) -> None:
        # The case this corpus is actually in.
        side: Notes = [("eighth", (("1", "continue"),))] * 100
        after: Notes = [("eighth", (("1", "end"),))] * 5 + [("eighth", (("1", "continue"),))] * 95

        result = compare(side, after)

        self.assertFalse(result.materialization_needed)
        self.assertFalse(result.round_trip_safe)
        self.assertIn("skip materialization", result.verdict())
        self.assertIn("do not use a MuseScore round trip", result.verdict())


if __name__ == "__main__":
    unittest.main()
