"""Replace beam groups the head emitted that no engraver could draw.

The beam head predicts each note's vector independently, so nothing makes the sequence
coherent. Audited over a dumped prediction file, it emits an interior impossibility on
**27.3% of staves** - a group that begins and never ends, an end with no beginning, a beam
nested inside a beam - against 6.2% for the engraved reference under identical processing,
and 682 nested groups against the reference's 12.

This does not make the deterministic rule the default. That is a larger question: the rule
and the head fail on nearly disjoint notes, and swapping the default would need its
accuracy re-measured in place. What it does is narrower and cannot lose information: where
a group is *provably* undrawable, the head has told us nothing usable about it, so the rule
is strictly better than what is there.

Groups that validate are left exactly as decoded, including the ones the rule would
disagree with - those are the cases the head exists for, and a beam spanning a rest is the
clearest of them, something the rule cannot express at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.beam_validation import validate_voice
from homr.transformer.structured_notation import BeamLevelState
from homr.transformer.vocabulary import EncodedSymbol

#: Fine enough that every duration this vocabulary can express lands on an integer,
#: including a dotted 64th and the tuplet denominators.
DIVISIONS = 48

#: Levels the validator and the rule both reason over.
LEVELS = 4


@dataclass
class BeamRepair:
    """What the pass did, so a caller reports rather than infers."""

    staves: int = 0
    repaired_staves: int = 0
    notes_rewritten: int = 0

    def describe(self) -> str:
        return (
            f"beam repair: {self.repaired_staves:,} of {self.staves:,} staves carried an "
            f"undrawable group; {self.notes_rewritten:,} notes rewritten"
        )


def _duration_and_flags(rhythm: str) -> tuple[Fraction, int] | None:
    """(quarters, flags) for a note or rest token.

    The denominator counts how many fit in a whole note, so it is not restricted to
    powers of two - `note_12` is a triplet eighth. Sounded duration and written value
    diverge there: a triplet eighth lasts a third of a quarter and still carries one flag.
    Onsets need the first, beaming the second.
    """
    if not rhythm.startswith(("note_", "rest_")):
        return None
    body = rhythm.split("_", 1)[1]
    digits = ""
    for character in body:
        if character.isdigit():
            digits += character
        else:
            break
    if not digits or int(digits) <= 0:
        return None
    denominator = int(digits)
    duration = Fraction(4, denominator)
    written = 1
    while written * 2 <= denominator:
        written *= 2
    flags = 0
    value = written
    while value >= 8:
        flags += 1
        value //= 2
    for character in body[len(digits) :]:
        if character != ".":
            break
        duration += duration / 2
    return duration, flags


def _metre(staff: Sequence[EncodedSymbol]) -> tuple[int, int]:
    for symbol in staff:
        if symbol.rhythm.startswith("timeSignature"):
            body = symbol.rhythm.split("timeSignature", 1)[1].lstrip("_")
            beats, _, beat_type = body.partition("/")
            if beats.isdigit() and beat_type.isdigit():
                return int(beats), int(beat_type)
            if beat_type.isdigit():
                return 4, int(beat_type)
    return 4, 4


def _run_around(vectors: list[tuple[BeamLevelState, ...]], position: int) -> set[int]:
    """Every note of the beam run containing `position`, judged at level 1."""
    joined = {
        BeamLevelState.BEGIN,
        BeamLevelState.CONTINUE,
        BeamLevelState.END,
        BeamLevelState.FORWARD_HOOK,
        BeamLevelState.BACKWARD_HOOK,
    }

    def state(index: int) -> BeamLevelState:
        vector = vectors[index]
        return vector[0] if vector else BeamLevelState.NOT_APPLICABLE

    left = position
    while left > 0 and state(left) != BeamLevelState.BEGIN:
        if state(left - 1) not in joined or state(left - 1) == BeamLevelState.END:
            break
        left -= 1
    right = position
    while right < len(vectors) - 1 and state(right) != BeamLevelState.END:
        if state(right + 1) not in joined or state(right + 1) == BeamLevelState.BEGIN:
            break
        right += 1
    return set(range(left, right + 1))


def repair_beams(staff: Sequence[EncodedSymbol]) -> BeamRepair:
    """Rewrite undrawable beam groups in place, using the rule. Returns what it did."""
    report = BeamRepair()
    indices: list[int] = []
    notes: list[BeamableNote] = []
    vectors: list[tuple[BeamLevelState, ...]] = []
    onset = 0
    for index, symbol in enumerate(staff):
        parsed = _duration_and_flags(symbol.rhythm)
        if parsed is None:
            continue
        duration, flags = parsed
        notation = getattr(symbol, "notation", None)
        ticks = int(duration * DIVISIONS)
        if notation is not None:
            indices.append(index)
            notes.append(
                BeamableNote(
                    onset=onset,
                    duration=ticks,
                    flags=flags,
                    is_rest=symbol.rhythm.startswith("rest_"),
                )
            )
            vectors.append(tuple(notation.beam_levels))
        onset += ticks

    if not vectors:
        return report
    report.staves = 1

    findings = validate_voice(vectors, LEVELS)
    if findings.valid:
        return report

    # Only the notes inside a broken group are rewritten. A staff usually carries several
    # groups and most of them are fine; replacing the whole staff would discard correct
    # beaming - including the rest-spanning groups the rule cannot produce - to fix one.
    #
    # A finding names the offending note, not its group, and rewriting that note alone
    # leaves the rest of the group as it was: an unclosed BEGIN repaired to BEGIN is no
    # repair. So each finding is expanded to the run it belongs to - stopping at an END
    # going right and a BEGIN going left, because those bound a group, and walking merely
    # "joined" states would swallow the sound group next door.
    broken: set[int] = set()
    for position in set(
        findings.unopened
        + findings.unclosed
        + findings.nested
        + findings.hook_at_primary_level
        + findings.single_note_groups
    ):
        broken.update(_run_around(vectors, position))
    if not broken:
        return report

    beats, beat_type = _metre(staff)
    predicted = automatic_beams(
        notes, beat_divisions(beats, beat_type, DIVISIONS), wide_unit(beats, beat_type, DIVISIONS)
    )
    rewritten = 0
    for position, index in enumerate(indices):
        if position not in broken:
            continue
        symbol = staff[index]
        notation = getattr(symbol, "notation", None)
        if notation is None:
            continue
        replacement = tuple(predicted[position])
        if tuple(notation.beam_levels) == replacement:
            continue
        symbol.notation = replace(notation, beam_levels=replacement)
        rewritten += 1

    report.repaired_staves = 1 if rewritten else 0
    report.notes_rewritten = rewritten
    return report
