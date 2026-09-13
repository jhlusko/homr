"""Make every tie we write joinable, by enforcing the one rule that defines a tie.

A tie is not a curve that happens to look like a slur. `TieState`'s own docstring puts it
exactly: *"a tie joins two notations of one pitch into a single sounding note, while a slur
groups distinct pitches under one phrase."* Same pitch is a **necessary condition** - a tie
between two different pitches is not a close call, it is impossible.

The tie head predicts each note's state independently, so nothing makes the two ends of one
tie agree. Measured on `IMSLP183800-sys5-v1`, the decode wrote eight `<tie>` elements and
**not one of them paired**: three starts whose pitch never recurs, five stops with no start
of that pitch before them. A reader draws none of it, so a page that plainly carries three
ties renders with zero.

What this does and does not touch is settled by two measurements
(`training/omr_datasets/tie_baseline.py`, 797,487 notes):

    reference ties joining a repeat of the same pitch   26,844 / 27,287   98.4%
    adjacent same-pitch pairs the engraving ties        26,844 / 169,098  15.9%

The constraint is nearly exact, so it can be enforced. The converse is not - only one
repeated pitch in six is tied - so **whether** a tie exists stays the head's decision and is
never invented here. This pass only makes the head's own assertion drawable: where a start
has a legitimate partner the partner is marked, and where the pitch does not recur at all
the assertion could not have been right and is dropped.

A tie whose partner is in the next system is ordinary engraving and is left alone: it has
no partner *in this voice* and never will, the same way a beam open at a crop edge is not
an unclosed beam.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from homr.transformer.structured_notation import TieState, VoiceClass
from homr.transformer.vocabulary import EncodedSymbol

_OPENS = {TieState.START, TieState.START_AND_STOP}
_CLOSES = {TieState.STOP, TieState.START_AND_STOP}


@dataclass
class TieRepair:
    """What the pass did, so a caller reports rather than infers."""

    starts: int = 0
    already_paired: int = 0
    partner_marked: int = 0
    dropped_starts: int = 0
    dropped_stops: int = 0
    at_edge: int = 0

    @property
    def changed(self) -> int:
        return self.partner_marked + self.dropped_starts + self.dropped_stops

    def describe(self) -> str:
        return (
            f"tie repair: {self.starts:,} starts, {self.already_paired:,} already joined, "
            f"{self.partner_marked:,} partner marked, {self.dropped_starts:,} start(s) and "
            f"{self.dropped_stops:,} stop(s) dropped, {self.at_edge:,} continuing past the voice"
        )


@dataclass(frozen=True)
class _Note:
    index: int
    pitch: str
    staff: str
    chord: int
    is_rest: bool
    #: Which line within the staff, when the source stated one. A tie joins one pitch
    #: within one voice; with two voices flattened together the next chord on the staff
    #: is as likely to be the other voice's, and the partner cannot be found at all.
    #: UNKNOWN means the source said nothing, and then the staff is the finest division
    #: available - which is what every corpus written before voices existed carries.
    voice: VoiceClass = VoiceClass.UNKNOWN


def _read(staff: Sequence[EncodedSymbol]) -> list[_Note]:
    """Note-bearing symbols with a chord id, so a chord's members group together.

    The chord id matters for ties. A chord's members are consecutive, so "the next note"
    for a chord member is its own sibling - the wrong candidate entirely, since a tie
    joins one notehead to a notehead in the *next* chord.
    """
    notes: list[_Note] = []
    chord = 0
    for index, symbol in enumerate(staff):
        if symbol.rhythm == "chord":
            continue
        if not symbol.rhythm.startswith(("note", "rest")):
            continue
        previous = staff[index - 1].rhythm if index else ""
        if previous != "chord":
            chord += 1
        notation = symbol.notation
        notes.append(
            _Note(
                index,
                symbol.pitch,
                symbol.position,
                chord,
                symbol.rhythm.startswith("rest"),
                notation.voice if notation is not None else VoiceClass.UNKNOWN,
            )
        )
    return notes


def _partner(notes: list[_Note], position: int) -> int | None:
    """The index in `notes` of the note a tie here would join, or None.

    The partner sits in the next chord on this staff. A rest between them ends the
    sounding note, so nothing can be tied across it.
    """
    here = notes[position]

    def same_line(other: _Note) -> bool:
        """Same staff, and the same voice when both notes name one."""
        if other.staff != here.staff:
            return False
        if VoiceClass.UNKNOWN in (here.voice, other.voice):
            return True
        return other.voice == here.voice

    for later in range(position + 1, len(notes)):
        candidate = notes[later]
        if not same_line(candidate):
            continue
        if candidate.chord == here.chord:
            continue
        if candidate.is_rest:
            return None
        # A chord can hold the same pitch in more than one voice; any member with this
        # pitch is a legitimate partner.
        members = [
            other for other in notes[later:] if other.chord == candidate.chord and same_line(other)
        ]
        for member in members:
            if member.pitch == here.pitch:
                return notes.index(member)
        return None
    return None


def _set_tie(staff: Sequence[EncodedSymbol], index: int, state: TieState) -> None:
    notation = staff[index].notation
    if notation is not None and notation.tie != state:
        staff[index].notation = replace(notation, tie=state)


def repair_ties(staff: Sequence[EncodedSymbol]) -> TieRepair:
    """Make each tie the head asserted joinable, or remove it, in place."""
    report = TieRepair()
    notes = _read(staff)
    states: dict[int, TieState] = {}
    for note in notes:
        notation = staff[note.index].notation
        if notation is not None:
            states[note.index] = notation.tie
    if not states:
        return report

    wanted: dict[int, set[TieState]] = {}

    for position, note in enumerate(notes):
        state = states.get(note.index)
        if state is None or state not in _OPENS:
            continue
        partner = _partner(notes, position)
        if partner is None:
            # Either the pitch does not recur - the assertion could not have been right -
            # or the voice ends here and the tie continues into the next system.
            if position == len(notes) - 1:
                report.at_edge += 1
                wanted.setdefault(note.index, set()).add(TieState.START)
            else:
                report.dropped_starts += 1
            continue
        report.starts += 1
        partner_index = notes[partner].index
        if states.get(partner_index) in _CLOSES:
            report.already_paired += 1
        else:
            report.partner_marked += 1
        wanted.setdefault(note.index, set()).add(TieState.START)
        wanted.setdefault(partner_index, set()).add(TieState.STOP)

    # A stop nobody opened cannot be drawn. Keep one only where the previous chord holds
    # this pitch and the voice began before it - the mirror of the rule above.
    for position, note in enumerate(notes):
        if states.get(note.index) not in _CLOSES:
            continue
        if TieState.STOP in wanted.get(note.index, set()):
            continue
        if position == 0:
            report.at_edge += 1
            wanted.setdefault(note.index, set()).add(TieState.STOP)
        else:
            report.dropped_stops += 1

    for note in notes:
        if states.get(note.index) is None:
            continue
        marks = wanted.get(note.index, set())
        if marks == {TieState.START, TieState.STOP}:
            _set_tie(staff, note.index, TieState.START_AND_STOP)
        elif marks == {TieState.START}:
            _set_tie(staff, note.index, TieState.START)
        elif marks == {TieState.STOP}:
            _set_tie(staff, note.index, TieState.STOP)
        else:
            _set_tie(staff, note.index, TieState.NONE)
    return report
