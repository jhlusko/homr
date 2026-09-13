import unittest
from fractions import Fraction

from homr.beam_repair import _duration_and_flags, repair_beams
from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
)
from homr.transformer.vocabulary import EncodedSymbol

LEVELS = 6
N = BeamLevelState.NOT_APPLICABLE


def _note(rhythm: str, *levels: BeamLevelState) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm=rhythm, pitch="C4")
    padded = tuple(levels) + (N,) * (LEVELS - len(levels))
    symbol.notation = NoteNotation(
        beam_levels=padded,
        stem=StemDirection.UP,
        slurs=((SlurEvent.NONE, SlurSide.UNSPECIFIED),) * LEVELS,
        tie=TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )
    return symbol


def _time(beats: int = 4, beat_type: int = 4) -> EncodedSymbol:
    return EncodedSymbol(rhythm=f"timeSignature{beats}/{beat_type}")


def _levels(symbol: EncodedSymbol) -> tuple[BeamLevelState, ...]:
    return tuple(symbol.notation.beam_levels)


class TestBeamRepair(unittest.TestCase):
    def test_a_valid_staff_is_untouched(self) -> None:
        staff = [
            _time(),
            _note("note_8", BeamLevelState.BEGIN),
            _note("note_8", BeamLevelState.END),
        ]
        before = [_levels(s) for s in staff[1:]]
        report = repair_beams(staff)
        self.assertEqual(0, report.notes_rewritten)
        self.assertEqual([_levels(s) for s in staff[1:]], before)

    def test_a_group_that_never_closes_is_rewritten(self) -> None:
        staff = [
            _time(),
            _note("note_8", BeamLevelState.BEGIN),
            _note("note_8", BeamLevelState.CONTINUE),
        ]
        report = repair_beams(staff)
        self.assertEqual(1, report.repaired_staves)
        self.assertGreater(report.notes_rewritten, 0)
        # Whatever the rule chose, the result has to be drawable.
        self.assertNotEqual(BeamLevelState.CONTINUE, _levels(staff[2])[0])

    def test_a_rest_spanning_beam_survives(self) -> None:
        """The case the head exists for must not be repaired away.

        The rule cannot produce a beam across a rest - a rest ends a group by
        construction - so a pass that rewrote valid groups would delete exactly the
        capability this head is kept for.
        """
        staff = [
            _time(),
            _note("note_16", BeamLevelState.BEGIN, BeamLevelState.BEGIN),
            _note("rest_16", BeamLevelState.CONTINUE, BeamLevelState.CONTINUE),
            _note("note_16", BeamLevelState.END, BeamLevelState.END),
        ]
        before = [_levels(s) for s in staff[1:]]
        repair_beams(staff)
        self.assertEqual([_levels(s) for s in staff[1:]], before)

    def test_only_the_broken_group_is_touched(self) -> None:
        """A staff usually carries several groups and most are fine."""
        staff = [
            _time(),
            # A sound group.
            _note("note_8", BeamLevelState.BEGIN),
            _note("note_8", BeamLevelState.END),
            # An END with nothing open before it.
            _note("note_8", BeamLevelState.END),
        ]
        sound = [_levels(staff[1]), _levels(staff[2])]
        repair_beams(staff)
        self.assertEqual([_levels(staff[1]), _levels(staff[2])], sound)

    def test_the_run_stops_at_the_next_group(self) -> None:
        """An unopened run must not swallow the sound group that follows it.

        Walking merely "joined" states would: BEGIN is a joined state, so a right-walk
        with no boundary check runs straight into the next group and rewrites beaming
        that was correct.
        """
        staff = [
            _time(),
            _note("note_8", BeamLevelState.CONTINUE),  # unopened
            _note("note_8", BeamLevelState.BEGIN),  # a sound group starts here
            _note("note_8", BeamLevelState.END),
        ]
        sound = [_levels(staff[2]), _levels(staff[3])]
        repair_beams(staff)
        self.assertEqual([_levels(staff[2]), _levels(staff[3])], sound)

    def test_flag_count_follows_the_written_value(self) -> None:
        """A sixteenth carries two flags, a triplet eighth one.

        The rule beams by flag count, so getting this wrong silently changes which levels
        a group is allowed to have - and tuplets are where the written value stops
        matching the sounded duration.
        """
        from fractions import Fraction

        from homr.beam_repair import _duration_and_flags

        self.assertEqual((Fraction(1), 0), _duration_and_flags("note_4"))
        self.assertEqual((Fraction(1, 2), 1), _duration_and_flags("note_8"))
        self.assertEqual((Fraction(1, 4), 2), _duration_and_flags("note_16"))
        # A triplet eighth: a third of a quarter, still drawn with one flag.
        self.assertEqual((Fraction(1, 3), 1), _duration_and_flags("note_12"))
        # Dotted: half as much again.
        self.assertEqual((Fraction(3, 4), 1), _duration_and_flags("note_8."))
        self.assertIsNone(_duration_and_flags("barline"))

    def test_a_staff_with_no_notation_is_skipped(self) -> None:
        staff = [_time(), EncodedSymbol(rhythm="note_8", pitch="C4")]
        report = repair_beams(staff)
        self.assertEqual(0, report.staves)

    def test_triplet_durations_do_not_break_the_onset_grid(self) -> None:
        """`note_12` is a triplet eighth: a third of a quarter, drawn with one flag.

        A parser that only knew powers of two would drop these, and the group would be
        measured against the wrong beat.
        """
        staff = [
            _time(),
            _note("note_12", BeamLevelState.BEGIN),
            _note("note_12", BeamLevelState.CONTINUE),
            _note("note_12", BeamLevelState.END),
        ]
        before = [_levels(s) for s in staff[1:]]
        report = repair_beams(staff)
        self.assertEqual(1, report.staves)
        self.assertEqual(0, report.notes_rewritten)
        self.assertEqual([_levels(s) for s in staff[1:]], before)


if __name__ == "__main__":
    unittest.main()


class TestDurationParsing(unittest.TestCase):
    """The onset grid the rule beams against.

    A wrong duration does not produce a wrong beam on that note - it shifts every onset
    after it, so the rule groups the remainder of the staff against a beat structure the
    music does not have. Measured end to end, `note_8..` was the single most common
    rhythm among the notes this pass rewrote from right to wrong (44 of 227) and appeared
    zero times among the ones it rewrote from wrong to right.
    """

    def test_a_single_dot_adds_half(self) -> None:
        self.assertEqual((Fraction(3, 4), 1), _duration_and_flags("note_8."))

    def test_a_second_dot_adds_a_quarter_not_another_half(self) -> None:
        """`note_8..` is 1/2 + 1/4 + 1/8 quarters. Compounding gives 9/8, which is wrong."""
        self.assertEqual((Fraction(7, 8), 1), _duration_and_flags("note_8.."))
        self.assertEqual((Fraction(7, 4), 0), _duration_and_flags("note_4.."))

    def test_a_triplet_keeps_its_written_flag_count(self) -> None:
        self.assertEqual((Fraction(1, 3), 1), _duration_and_flags("note_12"))
        self.assertEqual((Fraction(1, 6), 2), _duration_and_flags("note_24"))

    def test_a_grace_note_takes_no_metric_time(self) -> None:
        """It is played inside its neighbour's time, so it must not advance the grid."""
        parsed = _duration_and_flags("note_32G")
        assert parsed is not None  # noqa: S101
        self.assertEqual(Fraction(0), parsed[0])

    def test_a_rest_parses_like_a_note(self) -> None:
        self.assertEqual((Fraction(1, 2), 1), _duration_and_flags("rest_8"))

    def test_anything_that_is_not_a_note_or_rest_is_none(self) -> None:
        for rhythm in ("barline", "clef_G2", "timeSignature/4"):
            self.assertIsNone(_duration_and_flags(rhythm), rhythm)
