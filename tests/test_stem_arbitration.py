import unittest

from homr.stem_arbitration import HEAD_CONFIDENCE_THRESHOLD, arbitrate_stems
from homr.transformer.structured_decode import STEM_HEAD, HeadChoice
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


def _notation(first: BeamLevelState, stem: StemDirection) -> NoteNotation:
    return NoteNotation(
        beam_levels=(first,) + (BeamLevelState.NOT_APPLICABLE,) * (LEVELS - 1),
        stem=stem,
        slurs=((SlurEvent.NONE, SlurSide.UNSPECIFIED),) * LEVELS,
        tie=TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )


def _note(
    pitch: str, beam: BeamLevelState, stem: StemDirection, confidence: float
) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm="note_8", pitch=pitch)
    symbol.notation = _notation(beam, stem)
    symbol.structured_choices = (
        HeadChoice(head=STEM_HEAD, value=str(stem), probability=confidence),
    )
    return symbol


def _clef(name: str = "clef_G2") -> EncodedSymbol:
    return EncodedSymbol(rhythm=name)


class TestStemArbitration(unittest.TestCase):
    def test_a_confident_head_is_left_alone(self) -> None:
        """The rule is not a correction; above the threshold the head is the better source."""
        staff = [
            _clef(),
            # Both well below the middle line, so the rule would say UP.
            _note("C4", BeamLevelState.BEGIN, StemDirection.DOWN, 0.99),
            _note("D4", BeamLevelState.END, StemDirection.DOWN, 0.99),
        ]
        report = arbitrate_stems(staff)
        self.assertEqual(2, report.head_kept)
        self.assertEqual(0, report.changed)
        self.assertEqual(StemDirection.DOWN, staff[1].notation.stem)

    def test_an_unsure_head_is_replaced_by_the_group_direction(self) -> None:
        staff = [
            _clef(),
            _note("C4", BeamLevelState.BEGIN, StemDirection.DOWN, 0.4),
            _note("D4", BeamLevelState.END, StemDirection.DOWN, 0.4),
        ]
        report = arbitrate_stems(staff)
        self.assertEqual(2, report.rule_applied)
        self.assertEqual(2, report.changed)
        self.assertEqual(StemDirection.UP, staff[1].notation.stem)
        self.assertEqual(StemDirection.UP, staff[2].notation.stem)

    def test_the_whole_group_takes_the_extreme_notehead_s_direction(self) -> None:
        """One note near the middle does not get its own direction.

        This is the convention the rule exists to apply: the notehead furthest from the
        middle line decides, so a note that would individually point the other way still
        follows its group.
        """
        staff = [
            _clef(),
            # Six steps below the middle line: on its own this note says UP.
            _note("C4", BeamLevelState.BEGIN, StemDirection.UP, 0.1),
            # Eight above: further out, so it decides, and the group says DOWN.
            _note("C6", BeamLevelState.END, StemDirection.UP, 0.1),
        ]
        arbitrate_stems(staff)
        self.assertEqual(StemDirection.DOWN, staff[1].notation.stem)
        self.assertEqual(StemDirection.DOWN, staff[2].notation.stem)

    def test_the_clef_in_force_decides_the_middle_line(self) -> None:
        """The same pitch is above the middle line in bass clef and below it in treble."""
        treble = [
            _clef("clef_G2"),
            _note("D4", BeamLevelState.BEGIN, StemDirection.DOWN, 0.1),
            _note("E4", BeamLevelState.END, StemDirection.DOWN, 0.1),
        ]
        bass = [
            _clef("clef_F4"),
            _note("D4", BeamLevelState.BEGIN, StemDirection.UP, 0.1),
            _note("E4", BeamLevelState.END, StemDirection.UP, 0.1),
        ]
        arbitrate_stems(treble)
        arbitrate_stems(bass)
        self.assertEqual(StemDirection.UP, treble[1].notation.stem)
        self.assertEqual(StemDirection.DOWN, bass[1].notation.stem)

    def test_an_unbeamed_note_is_never_touched(self) -> None:
        """The rule only speaks about groups; a lone note keeps whatever the head said."""
        staff = [_clef(), _note("C4", BeamLevelState.FLAG, StemDirection.DOWN, 0.1)]
        report = arbitrate_stems(staff)
        self.assertEqual(0, report.notes)
        self.assertEqual(StemDirection.DOWN, staff[1].notation.stem)

    def test_a_group_that_never_ends_is_still_handled(self) -> None:
        """Heads emit sequences no engraver would draw; this must not raise."""
        staff = [
            _clef(),
            _note("C4", BeamLevelState.BEGIN, StemDirection.DOWN, 0.1),
            _note("D4", BeamLevelState.CONTINUE, StemDirection.DOWN, 0.1),
        ]
        report = arbitrate_stems(staff)
        self.assertEqual(2, report.notes)

    def test_a_one_note_group_is_not_a_group(self) -> None:
        """A BEGIN the head never closed leaves one note, and one note has no group.

        Applying a group direction to it would mean inventing a convention from a single
        notehead, which is just the pitch rule wearing a different hat - and the pitch
        rule is the 91% the arbitration exists to improve on.
        """
        staff = [_clef(), _note("C4", BeamLevelState.BEGIN, StemDirection.DOWN, 0.1)]
        report = arbitrate_stems(staff)
        self.assertEqual(0, report.notes)
        self.assertEqual(StemDirection.DOWN, staff[1].notation.stem)

    def test_a_symbol_without_notation_is_skipped(self) -> None:
        staff = [_clef(), EncodedSymbol(rhythm="note_8", pitch="C4")]
        self.assertEqual(0, arbitrate_stems(staff).notes)

    def test_the_threshold_is_the_measured_one(self) -> None:
        """0.9 is 27.28's swept value, tuned on one half and reported on the other."""
        self.assertEqual(0.9, HEAD_CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
