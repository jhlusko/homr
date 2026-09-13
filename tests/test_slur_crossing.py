"""Crossing slurs are re-paired; everything else is left exactly as decoded."""

import unittest
from dataclasses import replace

from homr.slur_crossing import EMPTY, Span, crosses, repair_crossings, spans_of
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

ABOVE, BELOW = SlurSide.ABOVE, SlurSide.BELOW


def _note(*slots: tuple[SlurEvent, SlurSide], position: str = "upper") -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm="note_4", pitch="G4", position=position)
    filled = list(slots) + [EMPTY] * (2 - len(slots))
    symbol.notation = NoteNotation(
        beam_levels=(BeamLevelState.NOT_APPLICABLE,) * 4,
        stem=StemDirection.UP,
        slurs=tuple(filled),
        tie=TieState.NONE,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )
    return symbol


def _plain() -> EncodedSymbol:
    return _note()


def _ranges(staff) -> set[tuple[int, int]]:
    return {(span.start, span.stop) for span in spans_of(staff)}


def _crossing_staff(side_two: SlurSide = ABOVE, position_two: str = "upper") -> list:
    """Slot 1 spans notes 0-2, slot 2 spans notes 1-3: interleaved."""
    return [
        _note((SlurEvent.START, ABOVE)),
        _note(EMPTY, (SlurEvent.START, side_two), position=position_two),
        _note((SlurEvent.STOP, ABOVE)),
        _note(EMPTY, (SlurEvent.STOP, side_two), position=position_two),
    ]


class TestDetection(unittest.TestCase):
    def test_interleaved_spans_cross(self) -> None:
        a = Span(start=0, stop=2, slot=0, side=ABOVE, staff="upper")
        b = Span(start=1, stop=3, slot=1, side=ABOVE, staff="upper")
        self.assertTrue(crosses(a, b))
        self.assertTrue(crosses(b, a))

    def test_a_phrase_inside_a_phrase_is_ordinary_engraving(self) -> None:
        outer = Span(start=0, stop=9, slot=0, side=ABOVE, staff="upper")
        inner = Span(start=2, stop=5, slot=1, side=ABOVE, staff="upper")
        self.assertFalse(crosses(outer, inner))

    def test_spans_that_follow_one_another_do_not_cross(self) -> None:
        first = Span(start=0, stop=3, slot=0, side=ABOVE, staff="upper")
        second = Span(start=4, stop=7, slot=0, side=ABOVE, staff="upper")
        self.assertFalse(crosses(first, second))

    def test_opposite_sides_never_intersect(self) -> None:
        above = Span(start=0, stop=2, slot=0, side=ABOVE, staff="upper")
        below = Span(start=1, stop=3, slot=1, side=BELOW, staff="upper")
        self.assertFalse(crosses(above, below))

    def test_different_staves_are_drawn_apart(self) -> None:
        upper = Span(start=0, stop=2, slot=0, side=ABOVE, staff="upper")
        lower = Span(start=1, stop=3, slot=1, side=ABOVE, staff="lower")
        self.assertFalse(crosses(upper, lower))

    def test_an_unspecified_side_is_not_judged(self) -> None:
        """Without a side there is no drawn geometry to call wrong."""
        a = Span(start=0, stop=2, slot=0, side=SlurSide.UNSPECIFIED, staff="upper")
        b = Span(start=1, stop=3, slot=1, side=SlurSide.UNSPECIFIED, staff="upper")
        self.assertFalse(crosses(a, b))


class TestSpanExtraction(unittest.TestCase):
    def test_an_unpaired_endpoint_is_not_a_span(self) -> None:
        """A slur continuing past the staff edge is correctly open there."""
        staff = [_note((SlurEvent.START, ABOVE)), _plain()]
        self.assertEqual([], spans_of(staff))

    def test_start_and_stop_closes_then_reopens_in_one_slot(self) -> None:
        staff = [
            _note((SlurEvent.START, ABOVE)),
            _note((SlurEvent.START_AND_STOP, ABOVE)),
            _note((SlurEvent.STOP, ABOVE)),
        ]
        self.assertEqual({(0, 1), (1, 2)}, _ranges(staff))


class TestRepair(unittest.TestCase):
    def test_a_crossing_pair_is_re_paired_to_nest(self) -> None:
        staff = _crossing_staff()
        report = repair_crossings(staff)
        self.assertEqual(1, report.repaired)
        self.assertEqual({(0, 3), (1, 2)}, _ranges(staff))

    def test_no_endpoint_moves(self) -> None:
        """The whole argument for this pass: only the pairing changes."""
        staff = _crossing_staff()
        before = [sorted(str(event) for event, _ in note.notation.slurs) for note in staff]
        repair_crossings(staff)
        after = [sorted(str(event) for event, _ in note.notation.slurs) for note in staff]
        self.assertEqual(before, after)

    def test_the_sides_are_untouched(self) -> None:
        staff = _crossing_staff()
        repair_crossings(staff)
        self.assertTrue(all(span.side == ABOVE for span in spans_of(staff)))

    def test_a_nested_pair_is_left_exactly_as_decoded(self) -> None:
        staff = [
            _note((SlurEvent.START, ABOVE)),
            _note(EMPTY, (SlurEvent.START, ABOVE)),
            _note(EMPTY, (SlurEvent.STOP, ABOVE)),
            _note((SlurEvent.STOP, ABOVE)),
        ]
        report = repair_crossings(staff)
        self.assertEqual(0, report.crossings)
        self.assertEqual({(0, 3), (1, 2)}, _ranges(staff))

    def test_opposite_sides_are_left_alone(self) -> None:
        staff = _crossing_staff(side_two=BELOW)
        self.assertEqual(0, repair_crossings(staff).crossings)
        self.assertEqual({(0, 2), (1, 3)}, _ranges(staff))

    def test_two_staves_are_left_alone(self) -> None:
        staff = _crossing_staff(position_two="lower")
        self.assertEqual(0, repair_crossings(staff).crossings)

    def test_an_occupied_target_slot_is_refused_rather_than_displaced(self) -> None:
        """A third span is involved; moving it would trade one crossing for another."""
        staff = _crossing_staff()
        # A third span opens in slot 1 on the note the later stop would have to move to.
        # It never closes, so it is not itself a span - but the slot is taken.
        blocked = staff[3]
        slots = list(blocked.notation.slurs)
        slots[0] = (SlurEvent.START, ABOVE)
        blocked.notation = replace(blocked.notation, slurs=tuple(slots))
        report = repair_crossings(staff)
        self.assertEqual(1, report.refused)
        self.assertEqual(0, report.repaired)

    def test_a_staff_with_no_slurs_reports_nothing(self) -> None:
        report = repair_crossings([_plain(), _plain()])
        self.assertEqual(0, report.spans)
        self.assertEqual(0, report.crossings)

    def test_the_result_has_no_crossings_left(self) -> None:
        staff = _crossing_staff()
        repair_crossings(staff)
        spans = spans_of(staff)
        self.assertFalse(any(crosses(a, b) for i, a in enumerate(spans) for b in spans[i + 1 :]))


if __name__ == "__main__":
    unittest.main()
