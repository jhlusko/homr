"""Slur side from the stem, where the head is unsure."""

import unittest

from homr.slur_side import HEAD_CONFIDENCE_THRESHOLD, choose_slur_sides
from homr.transformer.structured_decode import SLUR_SIDE_HEAD, HeadChoice
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


def _note(
    stem: StemDirection,
    event: SlurEvent = SlurEvent.START,
    side: SlurSide = SlurSide.ABOVE,
    confidence: float | None = None,
    slot: int = 0,
) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm="note_4", pitch="G4")
    slurs = [(SlurEvent.NONE, SlurSide.UNSPECIFIED), (SlurEvent.NONE, SlurSide.UNSPECIFIED)]
    slurs[slot] = (event, side)
    symbol.notation = NoteNotation(
        beam_levels=(BeamLevelState.NOT_APPLICABLE,) * 4,
        stem=stem,
        slurs=tuple(slurs),
        tie=TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )
    if confidence is not None:
        symbol.structured_choices = (
            HeadChoice(
                head=SLUR_SIDE_HEAD.format(slot=slot + 1),
                value=str(side),
                probability=confidence,
                alternatives=(),
            ),
        )
    return symbol


def _side(symbol: EncodedSymbol, slot: int = 0) -> SlurSide:
    return symbol.notation.slurs[slot][1]


class TestTheConvention(unittest.TestCase):
    def test_stems_up_puts_the_slur_below(self) -> None:
        note = _note(StemDirection.UP, side=SlurSide.ABOVE)
        report = choose_slur_sides([note])
        self.assertEqual(SlurSide.BELOW, _side(note))
        self.assertEqual(1, report.changed)

    def test_stems_down_puts_the_slur_above(self) -> None:
        note = _note(StemDirection.DOWN, side=SlurSide.BELOW)
        choose_slur_sides([note])
        self.assertEqual(SlurSide.ABOVE, _side(note))

    def test_a_side_the_rule_agrees_with_is_not_counted_as_a_change(self) -> None:
        note = _note(StemDirection.UP, side=SlurSide.BELOW)
        report = choose_slur_sides([note])
        self.assertEqual(1, report.rule_applied)
        self.assertEqual(0, report.changed)


class TestWhatItRefusesToTouch(unittest.TestCase):
    def test_a_slot_with_no_span_has_no_side_to_place(self) -> None:
        note = _note(StemDirection.UP, event=SlurEvent.NONE, side=SlurSide.UNSPECIFIED)
        report = choose_slur_sides([note])
        self.assertEqual(0, report.sides)
        self.assertEqual(SlurSide.UNSPECIFIED, _side(note))

    def test_an_unknown_stem_gives_the_rule_nothing_to_derive_from(self) -> None:
        """An invented side is worse than an uncertain one."""
        for stem in (StemDirection.UNKNOWN, StemDirection.NOT_APPLICABLE, StemDirection.NONE):
            with self.subTest(stem=stem):
                note = _note(stem, side=SlurSide.ABOVE)
                report = choose_slur_sides([note])
                self.assertEqual(0, report.sides)
                self.assertEqual(SlurSide.ABOVE, _side(note))

    def test_a_symbol_without_notation_is_skipped(self) -> None:
        bare = EncodedSymbol(rhythm="barline")
        choose_slur_sides([bare])
        self.assertIsNone(bare.notation)


class TestTheThreshold(unittest.TestCase):
    def test_a_confident_head_is_kept(self) -> None:
        note = _note(StemDirection.UP, side=SlurSide.ABOVE, confidence=0.99)
        report = choose_slur_sides([note])
        self.assertEqual(1, report.head_kept)
        self.assertEqual(SlurSide.ABOVE, _side(note))

    def test_an_unsure_head_is_overridden(self) -> None:
        note = _note(StemDirection.UP, side=SlurSide.ABOVE, confidence=0.5)
        report = choose_slur_sides([note])
        self.assertEqual(1, report.rule_applied)
        self.assertEqual(SlurSide.BELOW, _side(note))

    def test_the_threshold_is_inclusive(self) -> None:
        note = _note(StemDirection.UP, side=SlurSide.ABOVE, confidence=HEAD_CONFIDENCE_THRESHOLD)
        self.assertEqual(1, choose_slur_sides([note]).head_kept)

    def test_a_missing_confidence_falls_to_the_rule(self) -> None:
        """A checkpoint without the heads must still get the convention."""
        note = _note(StemDirection.UP, side=SlurSide.ABOVE)
        self.assertEqual(1, choose_slur_sides([note]).rule_applied)


class TestTheSecondSlot(unittest.TestCase):
    def test_a_span_in_slot_two_is_placed_by_its_own_confidence(self) -> None:
        note = _note(StemDirection.DOWN, side=SlurSide.BELOW, confidence=0.5, slot=1)
        choose_slur_sides([note])
        self.assertEqual(SlurSide.ABOVE, _side(note, slot=1))
        self.assertEqual(SlurSide.UNSPECIFIED, _side(note, slot=0))


if __name__ == "__main__":
    unittest.main()
