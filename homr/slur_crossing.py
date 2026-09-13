"""Re-pair slurs whose spans cross, which no engraver draws.

Two slurs may nest - one phrase inside another - and they may follow one another, but they
never cross. A crossing is not a choice between two readings of the music; it is a
construction that cannot be engraved, so wherever the decode produces one, the pairing is
wrong however plausible each endpoint looked on its own.

The heads predict each note's slur slots independently, which is exactly how a crossing
gets made: a start in slot 1 at note 3 and a stop in slot 1 at note 9, while slot 2 opens
at note 6 and closes at note 12. Every endpoint is individually reasonable and the four
together are not drawable.

**The correction invents nothing.** Both endpoints of both spans are already predicted, on
the notes the model chose, with the sides it chose. Only which start belongs to which stop
is wrong, and re-pairing an interleaved pair so that one nests inside the other moves no
endpoint at all - it exchanges the two stops' slots and nothing else. That is what makes
this different from the beam rule, which replaces a prediction with a guess.

Measured over 400 Lieder systems after the numbering fix (`homr/music_xml_generator.py`),
the decode draws 69 crossings on 9.50% of staves against the engraved reference's 0.25%.

Runs after `choose_slur_sides`, and must: slurs on opposite sides never intersect however
their endpoints order, so the side is part of the test and has to be the final one.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from homr.transformer.structured_notation import SlurEvent, SlurSide
from homr.transformer.vocabulary import EncodedSymbol

#: A slot holding no span at all.
EMPTY = (SlurEvent.NONE, SlurSide.UNSPECIFIED)

#: Give up rather than loop: each repair can expose another, but a staff with more than
#: this many interleavings is not a pairing slip, and rewriting it further is guesswork.
MAX_REPAIRS = 8


@dataclass(frozen=True)
class Span:
    """One slur, as the notation records it."""

    start: int
    stop: int
    slot: int
    side: SlurSide
    staff: str


@dataclass
class CrossingRepair:
    """What the pass did, so a caller reports rather than infers."""

    spans: int = 0
    crossings: int = 0
    repaired: int = 0
    refused: int = 0

    def describe(self) -> str:
        text = (
            f"slur crossing: {self.spans:,} spans, {self.crossings:,} crossing pair(s), "
            f"{self.repaired:,} re-paired"
        )
        return text + (f", {self.refused:,} left alone" if self.refused else "")


def _notes(staff: Sequence[EncodedSymbol]) -> list[int]:
    return [i for i, s in enumerate(staff) if s.rhythm.startswith(("note", "rest"))]


def spans_of(staff: Sequence[EncodedSymbol]) -> list[Span]:
    """Every paired span in the staff, per slot.

    An endpoint with no partner is not a span and is not returned: a slur continuing past
    the edge of the staff is correctly open there, and inventing a partner for it would
    create the very thing this module removes.
    """
    indices = _notes(staff)
    width = max(
        (len(s.notation.slurs) for s in staff if s.notation is not None),
        default=0,
    )
    found: list[Span] = []
    for slot in range(width):
        open_at: tuple[int, SlurSide, str] | None = None
        for position, index in enumerate(indices):
            notation = staff[index].notation
            if notation is None or slot >= len(notation.slurs):
                continue
            event, side = notation.slurs[slot]
            if event in (SlurEvent.STOP, SlurEvent.START_AND_STOP) and open_at is not None:
                start, start_side, staff_name = open_at
                found.append(Span(start, position, slot, start_side, staff_name))
                open_at = None
            if event in (SlurEvent.START, SlurEvent.START_AND_STOP):
                open_at = (position, side, staff[index].position)
    return found


def crosses(one: Span, other: Span) -> bool:
    """Whether these two spans cross as drawn.

    Three things have to hold. They must be on the same staff, because two staves are
    drawn apart. They must be on the same side, because a slur above and a slur below
    never intersect however their endpoints order. And they must interleave - one span
    beginning inside the other and ending outside it. Nesting is not crossing: a phrase
    inside a phrase is ordinary engraving.
    """
    if one.staff != other.staff:
        return False
    if one.side != other.side or one.side == SlurSide.UNSPECIFIED:
        return False
    first, second = sorted((one, other), key=lambda span: (span.start, span.stop))
    return first.start < second.start < first.stop < second.stop


def _swap_stops(staff: Sequence[EncodedSymbol], first: Span, second: Span) -> bool:
    """Exchange the two spans' stop slots, so they nest instead of crossing.

    `first` begins before `second` and ends before it. Giving `first`'s start the later
    stop and `second`'s start the earlier one turns the interleaving into `second` nested
    inside `first`. Both stops keep their notes and their sides; only the slot each sits
    in changes.

    Refuses when either target slot is already carrying something on that note - there is
    a third span involved, and displacing it would trade one crossing for another.
    """
    indices = _notes(staff)
    first_note, second_note = staff[indices[first.stop]], staff[indices[second.stop]]
    if first_note.notation is None or second_note.notation is None:
        return False

    first_slots, second_slots = list(first_note.notation.slurs), list(second_note.notation.slurs)
    if second.slot >= len(first_slots) or first.slot >= len(second_slots):
        return False
    if first_slots[second.slot] != EMPTY or second_slots[first.slot] != EMPTY:
        return False

    first_slots[second.slot] = first_slots[first.slot]
    first_slots[first.slot] = EMPTY
    second_slots[first.slot] = second_slots[second.slot]
    second_slots[second.slot] = EMPTY

    first_note.notation = replace(first_note.notation, slurs=tuple(first_slots))
    second_note.notation = replace(second_note.notation, slurs=tuple(second_slots))
    return True


def repair_crossings(staff: Sequence[EncodedSymbol]) -> CrossingRepair:
    """Re-pair every crossing pair of slurs in the staff, in place.

    Repeats because one repair can expose another: the spans are re-read after each
    change rather than repaired from a stale list, which is the difference between fixing
    a staff and rewriting one.
    """
    report = CrossingRepair()
    for _attempt in range(MAX_REPAIRS):
        spans = spans_of(staff)
        report.spans = len(spans)
        crossing = next(
            (
                (one, two)
                for index, one in enumerate(spans)
                for two in spans[index + 1 :]
                if crosses(one, two)
            ),
            None,
        )
        if crossing is None:
            return report
        report.crossings += 1
        first, second = sorted(crossing, key=lambda span: (span.start, span.stop))
        if _swap_stops(staff, first, second):
            report.repaired += 1
        else:
            report.refused += 1
            return report
    return report
