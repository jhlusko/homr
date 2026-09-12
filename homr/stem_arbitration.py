"""Choose each note's stem direction between the head's prediction and a rule.

27.27 measured the two sources over the 111,229 notes the stem head is scored on:

    pitch alone                     91.0%
    the trained stem head           94.3%     25,137 parameters
    grouped by predicted beams      94.4%     no parameters

Equal totals, and 27.28's crosstab shows they are not the same 94%: the head rescues
4,491 of the rule's mistakes, the rule rescues 4,506 of the head's, and only 1,690 notes
defeat both. Choosing per note from the head's own confidence was measured at **95.92%**,
against 94.4% for either alone, with the head used on 82.7% of notes. That is what this
applies.

**The rule is not a fallback for a missing head; it is better than the head where the head
is unsure.** So the arbitration runs whenever both are available, not only when the heads
are absent.

Positions come from the pitch and the clef in force, which is why this is a post-decode
pass over a staff rather than part of `structured_decode`: the heads see one note's hidden
state and nothing about where that note sits on the staff.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from homr.transformer.structured_decode import STEM_HEAD
from homr.transformer.structured_notation import BeamLevelState, StemDirection
from homr.transformer.vocabulary import EncodedSymbol

#: The head is trusted at or above this. Measured, not chosen: 27.28 swept it and reports
#: 95.92% at 0.9 against 95.60% for combining signals and 94.41% for the rule alone. The
#: sweep was tuned on half the staves and reported on the other half, so this number is
#: not fitted to what it is quoted against.
HEAD_CONFIDENCE_THRESHOLD = 0.9

_STEPS = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
#: What pitch each clef sign fixes to its own line.
_CLEF_PITCH = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}
_DEFAULT_CLEF = ("G", 2)

_JOINED = {
    BeamLevelState.BEGIN,
    BeamLevelState.CONTINUE,
    BeamLevelState.END,
    BeamLevelState.FORWARD_HOOK,
    BeamLevelState.BACKWARD_HOOK,
}


def _diatonic(step: str, octave: int) -> int:
    return octave * 7 + _STEPS.get(step.strip().upper(), 0)


def _middle_line(sign: str, line: int) -> int:
    """The diatonic index on the staff's middle line.

    A clef fixes one pitch to one line, the middle line is line 3, and consecutive lines
    are two diatonic steps apart.
    """
    step, octave = _CLEF_PITCH.get(sign.strip().upper(), ("G", 4))
    return _diatonic(step, octave) + 2 * (3 - line)


def _parse_clef(rhythm: str) -> tuple[str, int] | None:
    """`clef_G2` -> (G, 2). Anything else is not a clef."""
    if not rhythm.startswith("clef_"):
        return None
    body = rhythm[len("clef_") :]
    if len(body) < 2 or not body[1:].isdigit():
        return None
    return body[0], int(body[1:])


def _parse_pitch(pitch: str) -> tuple[str, int] | None:
    """`G4` or `Bb3` -> (step, octave). Accidentals are irrelevant to staff position."""
    text = pitch.strip()
    if len(text) < 2 or text[0].upper() not in _STEPS:
        return None
    octave_text = text[1:].lstrip("#b-")
    digits = "".join(
        character for character in octave_text if character.isdigit() or character == "-"
    )
    if not digits.lstrip("-").isdigit():
        return None
    return text[0], int(digits)


def _groups_from_beams(vectors: list[tuple[BeamLevelState, ...]]) -> list[list[int]]:
    """Note indices split into beam groups, using level 1 only.

    Level 1 carries the outer beam, which is what sets the direction; deeper levels
    subdivide a group that has already chosen one. Written to survive vectors a head can
    emit but an engraver would not: a BEGIN while a group is open closes the previous one
    rather than nesting, and a flag or an inapplicable level ends the run, because a
    flagged note cannot be inside a beam.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for index, vector in enumerate(vectors):
        state = vector[0] if vector else BeamLevelState.NOT_APPLICABLE
        if state == BeamLevelState.BEGIN:
            if current:
                groups.append(current)
            current = [index]
        elif state in {BeamLevelState.CONTINUE, BeamLevelState.END} and current:
            current.append(index)
            if state == BeamLevelState.END:
                groups.append(current)
                current = []
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [group for group in groups if len(group) > 1]


@dataclass
class Arbitration:
    """What the pass did, so a caller can report it rather than infer it."""

    notes: int = 0
    head_kept: int = 0
    rule_applied: int = 0
    changed: int = 0

    def describe(self) -> str:
        return (
            f"stem arbitration: {self.notes:,} notes, head kept {self.head_kept:,}, "
            f"rule applied {self.rule_applied:,}, direction changed {self.changed:,}"
        )


def _confidence(symbol: EncodedSymbol) -> float | None:
    for choice in getattr(symbol, "structured_choices", ()) or ():
        if choice.head == STEM_HEAD:
            return choice.probability
    return None


def arbitrate_stems(
    staff: Sequence[EncodedSymbol], threshold: float = HEAD_CONFIDENCE_THRESHOLD
) -> Arbitration:
    """Replace low-confidence head stems with the beam-group rule, in place.

    Returns what it did. A symbol with no notation, no pitch, or no beam group is left
    exactly as decoded - the rule has nothing to say about a note that is not in a group,
    and inventing a direction there would be worse than the head's guess.
    """
    report = Arbitration()
    clef = _DEFAULT_CLEF
    middle = _middle_line(*clef)

    notes: list[int] = []
    positions: dict[int, int] = {}
    vectors: list[tuple[BeamLevelState, ...]] = []
    for index, symbol in enumerate(staff):
        parsed_clef = _parse_clef(symbol.rhythm)
        if parsed_clef is not None:
            clef = parsed_clef
            middle = _middle_line(*clef)
            continue
        notation = getattr(symbol, "notation", None)
        if notation is None:
            continue
        parsed_pitch = _parse_pitch(symbol.pitch)
        if parsed_pitch is None:
            continue
        notes.append(index)
        positions[index] = _diatonic(*parsed_pitch) - middle
        vectors.append(tuple(notation.beam_levels))

    for group in _groups_from_beams(vectors):
        # One direction for the group, from the notehead furthest from the middle line -
        # which is what an engraver does, and what the rule was measured as.
        extreme = max((positions[notes[member]] for member in group), key=abs)
        direction = StemDirection.DOWN if extreme >= 0 else StemDirection.UP
        for member in group:
            index = notes[member]
            symbol = staff[index]
            report.notes += 1
            confidence = _confidence(symbol)
            if confidence is not None and confidence >= threshold:
                report.head_kept += 1
                continue
            report.rule_applied += 1
            notation = symbol.notation
            if notation is not None and notation.stem != direction:
                report.changed += 1
                symbol.notation = replace(notation, stem=direction)
    return report
