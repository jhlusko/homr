"""
Extract structured beam, stem and slur labels from MusicXML.

Read the original MusicXML, never the cleaned copy: cleaning flattens slur numbering to
1 and drops placement, so two of the three things extracted here would be silently
destroyed.

What this deliberately does not do is repair anything. Unmatched slur ends, slot
overflow and notes whose beaming is ambiguous are reported as findings and left alone,
because an output that parses cleanly can still be musically wrong, and a label pipeline
that quietly fixes its inputs produces training data nobody can audit.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from homr.transformer.structured_notation import (
    MAX_BEAM_LEVELS,
    MAX_SLUR_SLOTS,
    BeamLevelState,
    SlurEvent,
    SlurSide,
    StemDirection,
    applicable_beam_levels,
    empty_beam_levels,
    empty_slur_slots,
)

_BEAM_STATES = {
    "begin": BeamLevelState.BEGIN,
    "continue": BeamLevelState.CONTINUE,
    "end": BeamLevelState.END,
    "forward hook": BeamLevelState.FORWARD_HOOK,
    "backward hook": BeamLevelState.BACKWARD_HOOK,
}

_STEMS = {
    "up": StemDirection.UP,
    "down": StemDirection.DOWN,
    "none": StemDirection.NONE,
    "double": StemDirection.DOUBLE,
}

_SIDES = {
    "above": SlurSide.ABOVE,
    "over": SlurSide.ABOVE,
    "below": SlurSide.BELOW,
    "under": SlurSide.BELOW,
}


@dataclass(frozen=True)
class NoteNotation:
    """Structured notation for one note, in the order the part reads."""

    beam_levels: tuple[BeamLevelState, ...]
    stem: StemDirection
    slurs: tuple[tuple[SlurEvent, SlurSide], ...]

    def active_beam_levels(self) -> int:
        return sum(1 for state in self.beam_levels if state != BeamLevelState.NOT_APPLICABLE)


@dataclass
class Findings:
    """Everything the source got wrong or left ambiguous, counted rather than fixed."""

    #: A slur stop with no open span in that source identity.
    unmatched_stops: int = 0
    #: A span still open when the part ended.
    unclosed_starts: int = 0
    #: A start that found no free canonical slot, so it carries no label.
    slot_overflow: int = 0
    #: A source identity reopened while already open.
    duplicate_starts: int = 0
    #: A note whose duration carries flags but which has no <beam> at all. Before beam
    #: materialisation this is ambiguous - automatically beamed, or deliberately
    #: unbeamed - so it is counted, and callers that need exact beam labels should treat
    #: a nonzero count as "materialisation has not run".
    ambiguous_beaming: int = 0
    #: A beam element above the level the note's duration allows.
    beams_above_flag_depth: int = 0
    notes: int = 0
    messages: list[str] = field(default_factory=list)

    def add(self, other: "Findings") -> None:
        self.unmatched_stops += other.unmatched_stops
        self.unclosed_starts += other.unclosed_starts
        self.slot_overflow += other.slot_overflow
        self.duplicate_starts += other.duplicate_starts
        self.ambiguous_beaming += other.ambiguous_beaming
        self.beams_above_flag_depth += other.beams_above_flag_depth
        self.notes += other.notes
        self.messages.extend(other.messages)

    @property
    def clean(self) -> bool:
        return not (
            self.unmatched_stops
            or self.unclosed_starts
            or self.slot_overflow
            or self.duplicate_starts
            or self.beams_above_flag_depth
        )


def _beam_levels(note: ET.Element, findings: Findings) -> tuple[BeamLevelState, ...]:
    note_type = note.findtext("type")
    applicable = applicable_beam_levels(note_type)
    states = list(empty_beam_levels())
    written = 0
    for beam in note.findall("beam"):
        number = beam.get("number")
        if number is None or not number.isdigit():
            continue
        level = int(number)
        state = _BEAM_STATES.get((beam.text or "").strip().lower())
        if state is None:
            continue
        if level > applicable or level > MAX_BEAM_LEVELS:
            # A beam deeper than the duration allows cannot be rendered; record it and
            # drop it rather than writing a state the note cannot carry.
            findings.beams_above_flag_depth += 1
            continue
        states[level - 1] = state
        written += 1
    # Levels the duration supports but the source did not write are flags, not beams.
    for level in range(applicable):
        if states[level] == BeamLevelState.NOT_APPLICABLE:
            states[level] = BeamLevelState.FLAG
    if applicable and written == 0 and note.find("rest") is None:
        findings.ambiguous_beaming += 1
    return tuple(states)


def _stem(note: ET.Element) -> StemDirection:
    if note.find("rest") is not None:
        return StemDirection.NOT_APPLICABLE
    text = note.findtext("stem")
    if text is None:
        return StemDirection.UNKNOWN
    return _STEMS.get(text.strip().lower(), StemDirection.UNKNOWN)


def _side(slur: ET.Element) -> SlurSide:
    raw = slur.get("placement") or slur.get("orientation")
    if raw is None:
        return SlurSide.UNSPECIFIED
    return _SIDES.get(raw.strip().lower(), SlurSide.UNSPECIFIED)


class _SlurSlots:
    """Canonical slot assignment for one voice.

    MusicXML slur numbers identify paired elements within a document; they are not
    semantic labels, and the same number is reused freely once a span closes. So spans
    are mapped onto canonical slots by their own lifetime: a start takes the lowest free
    slot, its stop releases it, and the source number is only used to decide which open
    span a stop belongs to.
    """

    def __init__(self) -> None:
        #: source slur number -> canonical slot, for spans currently open
        self._open: dict[str, int] = {}

    def _free_slot(self) -> int | None:
        used = set(self._open.values())
        return next((slot for slot in range(MAX_SLUR_SLOTS) if slot not in used), None)

    def apply(
        self, slurs: list[ET.Element], findings: Findings
    ) -> tuple[tuple[SlurEvent, SlurSide], ...]:
        slots = list(empty_slur_slots())

        def record(slot: int, event: SlurEvent, side: SlurSide) -> None:
            existing_event, existing_side = slots[slot]
            if existing_event == SlurEvent.STOP and event == SlurEvent.START:
                # One note closing a span and opening another in the same slot is the
                # case START_AND_STOP exists for.
                slots[slot] = (SlurEvent.START_AND_STOP, side or existing_side)
            elif existing_event == SlurEvent.START and event == SlurEvent.STOP:
                slots[slot] = (SlurEvent.START_AND_STOP, existing_side)
            else:
                slots[slot] = (event, side)

        # Stops first, so a slot released on this note can be reused by a start on it.
        ordered = sorted(slurs, key=lambda s: 0 if s.get("type") == "stop" else 1)
        for slur in ordered:
            kind = (slur.get("type") or "").strip().lower()
            number = slur.get("number") or "1"
            if kind == "stop":
                slot = self._open.pop(number, None)
                if slot is None:
                    findings.unmatched_stops += 1
                    continue
                record(slot, SlurEvent.STOP, _side(slur))
            elif kind == "start":
                if number in self._open:
                    findings.duplicate_starts += 1
                    continue
                slot = self._free_slot()
                if slot is None:
                    findings.slot_overflow += 1
                    continue
                self._open[number] = slot
                record(slot, SlurEvent.START, _side(slur))
            elif kind == "continue":
                slot = self._open.get(number)
                if slot is None:
                    findings.unmatched_stops += 1
                    continue
                record(slot, SlurEvent.CONTINUE, _side(slur))
        return tuple(slots)

    def close(self, findings: Findings) -> None:
        findings.unclosed_starts += len(self._open)
        self._open.clear()


def parse_part(part: ET.Element) -> tuple[list[NoteNotation], Findings]:
    """Structured notation for every note of one <part>, in document order.

    Slur slots are tracked per voice, because two voices on a staff open and close their
    own spans independently and sharing one slot pool across them would make each voice's
    slots depend on the other's.
    """
    findings = Findings()
    result: list[NoteNotation] = []
    voices: dict[str, _SlurSlots] = {}
    for note in part.iter("note"):
        findings.notes += 1
        voice = note.findtext("voice") or "1"
        slots = voices.setdefault(voice, _SlurSlots())
        slurs = [
            slur for notations in note.findall("notations") for slur in notations.findall("slur")
        ]
        result.append(
            NoteNotation(
                beam_levels=_beam_levels(note, findings),
                stem=_stem(note),
                slurs=slots.apply(slurs, findings),
            )
        )
    for slots in voices.values():
        slots.close(findings)
    return result, findings


def parse_score(root: ET.Element) -> tuple[dict[str, list[NoteNotation]], Findings]:
    """Structured notation for every part of a score-partwise document."""
    findings = Findings()
    parts: dict[str, list[NoteNotation]] = {}
    for index, part in enumerate(root.findall("part")):
        notes, part_findings = parse_part(part)
        parts[part.get("id") or f"P{index + 1}"] = notes
        findings.add(part_findings)
    return parts, findings
