"""A slur number must be unique across every span open at that moment.

MusicXML pairs a slur by its `number`. `_slur_number` used to return the sidecar's slot
directly, and a slot is unique only within one note - so on a grand staff, slot 1 on the
upper staff and slot 1 on the lower staff both became `number="1"` and the reader was
handed two starts and two stops with no way to tell which belonged to which.

Measured over 400 Lieder systems before the fix: 315 colliding spans on 26% of engraved
staves and 250 on 22% of predicted ones, with a reader pairing them in the obvious order
drawing 131 crossing slurs on 15.75% of our staves against the reference's one.
"""

import unittest
import xml.etree.ElementTree as ET
from fractions import Fraction

from homr.music_xml_generator import ConversionState, build_slurs
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
    slur_token: str = "_",
    slots: tuple[tuple[SlurEvent, SlurSide], ...] = (),
    position: str = "upper",
) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm="note_4", pitch="G4", slur=slur_token, position=position)
    if slots:
        symbol.notation = NoteNotation(
            beam_levels=(BeamLevelState.NOT_APPLICABLE,) * 4,
            stem=StemDirection.UP,
            slurs=slots,
            tie=TieState.NONE,
            dynamic=DynamicMark.NONE,
            advance=AdvanceClass.NOT_APPLICABLE,
        )
    return symbol


NONE = (SlurEvent.NONE, SlurSide.UNSPECIFIED)


def _emit(notes: list[EncodedSymbol]) -> list[tuple[str, str]]:
    """(type, number) for every slur element the notes produce, in order."""
    state = ConversionState(division=4, nominator=Fraction(1))
    written = []
    for symbol in notes:
        element = ET.Element("note")
        build_slurs(element, symbol, state)
        for slur in element.findall("./notations/slur"):
            written.append((slur.get("type") or "", slur.get("number") or ""))
    return written


def _open_at_all_times(written: list[tuple[str, str]]) -> bool:
    """No number is ever opened twice without closing in between."""
    live: set[str] = set()
    for kind, number in written:
        if kind == "start":
            if number in live:
                return False
            live.add(number)
        else:
            live.discard(number)
    return True


class TestConcurrentSpansGetDistinctNumbers(unittest.TestCase):
    def test_two_staves_opening_their_own_slot_one_do_not_collide(self) -> None:
        """The grand-staff case: 220 of 315 measured collisions were this."""
        written = _emit(
            [
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE), "upper"),
                _note("slurStart", ((SlurEvent.START, SlurSide.BELOW), NONE), "lower"),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.BELOW), NONE), "lower"),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE), "upper"),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)
        self.assertNotEqual(written[0][1], written[1][1])

    def test_the_stop_takes_the_number_its_own_start_was_given(self) -> None:
        """Exact pairing, which is what the slots exist for."""
        written = _emit(
            [
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE), "upper"),
                _note("slurStart", ((SlurEvent.START, SlurSide.BELOW), NONE), "lower"),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE), "upper"),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.BELOW), NONE), "lower"),
            ]
        )
        upper_start, lower_start = written[0][1], written[1][1]
        self.assertEqual(upper_start, written[2][1])
        self.assertEqual(lower_start, written[3][1])

    def test_two_slots_on_one_staff_stay_distinct(self) -> None:
        written = _emit(
            [
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("slurStart", (NONE, (SlurEvent.START, SlurSide.ABOVE))),
                _note("slurStop", (NONE, (SlurEvent.STOP, SlurSide.ABOVE))),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)

    def test_indistinguishable_spans_still_get_distinct_numbers(self) -> None:
        """Two spans the sidecar files identically - 95 of 315 measured collisions.

        The sidecar cannot say which stop closes which, so the pairing falls back to
        last-opened. The numbers must still not collide, which is what makes the output
        readable at all.
        """
        written = _emit(
            [
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)

    def test_a_number_is_reused_once_its_span_closes(self) -> None:
        written = _emit(
            [
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
            ]
        )
        self.assertEqual("1", written[0][1])
        self.assertEqual("1", written[2][1])


class TestTheSidecarOnlyPath(unittest.TestCase):
    """Notes whose flat slur field says nothing but whose sidecar does."""

    def test_two_staves_do_not_collide_there_either(self) -> None:
        written = _emit(
            [
                _note("_", ((SlurEvent.START, SlurSide.ABOVE), NONE), "upper"),
                _note("_", ((SlurEvent.START, SlurSide.BELOW), NONE), "lower"),
                _note("_", ((SlurEvent.STOP, SlurSide.BELOW), NONE), "lower"),
                _note("_", ((SlurEvent.STOP, SlurSide.ABOVE), NONE), "upper"),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)

    def test_start_and_stop_closes_before_it_reopens(self) -> None:
        written = _emit(
            [
                _note("_", ((SlurEvent.START, SlurSide.ABOVE), NONE)),
                _note("_", ((SlurEvent.START_AND_STOP, SlurSide.ABOVE), NONE)),
                _note("_", ((SlurEvent.STOP, SlurSide.ABOVE), NONE)),
            ]
        )
        self.assertEqual([("start", "1"), ("stop", "1"), ("start", "1"), ("stop", "1")], written)


class TestWithoutASidecar(unittest.TestCase):
    """A checkpoint with no structured heads must behave exactly as it did."""

    def test_the_stack_still_nests(self) -> None:
        written = _emit(
            [_note("slurStart"), _note("slurStart"), _note("slurStop"), _note("slurStop")]
        )
        self.assertEqual([("start", "1"), ("start", "2"), ("stop", "2"), ("stop", "1")], written)

    def test_a_stop_with_nothing_open_still_emits(self) -> None:
        """The defect stays visible in the output rather than vanishing."""
        self.assertEqual([("stop", "1")], _emit([_note("slurStop")]))


if __name__ == "__main__":
    unittest.main()


class TestPairingPrefersTheSameStaff(unittest.TestCase):
    """A stop belongs to a span on its own staff unless nothing there is open.

    Cross-staff slurs are real in piano writing, so the last resort stays - but it is a
    last resort. On IMSLP183800-sys5-v1 the decode produced 7 spans and 4 of them paired
    across the grand staff, on a page with no cross-staff slur at all.
    """

    def test_a_stop_takes_the_span_open_on_its_own_staff(self) -> None:
        """Neither endpoint carries a slot, so this is the fallback path alone."""
        written = _emit(
            [
                _note("slurStart", position="upper"),
                _note("slurStart", position="lower"),
                _note("slurStop", position="upper"),
                _note("slurStop", position="lower"),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)
        # The upper stop must take the upper start's number, not the newer lower one.
        self.assertEqual(written[0][1], written[2][1])
        self.assertEqual(written[1][1], written[3][1])

    def test_it_still_crosses_staves_when_nothing_else_is_open(self) -> None:
        """A genuine cross-staff slur must still pair rather than be dropped."""
        written = _emit([_note("slurStart", position="upper"), _note("slurStop", position="lower")])
        self.assertEqual([("start", "1"), ("stop", "1")], written)

    def test_a_key_resolved_by_the_fallback_does_not_go_stale(self) -> None:
        """The bug behind the last two cross-staff pairs.

        A stop that resolves by staff rather than by its key left the key still holding
        that number, and the next stop filed under the key popped a span that had closed
        long ago - on whichever staff owned it.
        """
        written = _emit(
            [
                # Opens under key (1, 1).
                _note("slurStart", ((SlurEvent.START, SlurSide.ABOVE), NONE), "upper"),
                # Closes it by the staff fallback - this note carries no slot.
                _note("slurStop", position="upper"),
                # A fresh span on the lower staff.
                _note("slurStart", position="lower"),
                # Filed under key (1, 1) again: must not reach back for the closed span.
                _note("slurStop", ((SlurEvent.STOP, SlurSide.ABOVE), NONE), "upper"),
            ]
        )
        self.assertTrue(_open_at_all_times(written), written)
