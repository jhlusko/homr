"""
The beaming a deterministic engraver would produce from duration and meter alone.

This is the baseline the beam heads have to beat. Most beaming is not a choice: given a
metre and a run of short notes, an engraver beams them to the beat, and any program can
reconstruct that without looking at the page. A head that only reproduces those groups
has learned nothing that was not already derivable, so the number worth reporting is how
often it gets the *exceptions* right - the places where the engraver did something the
rule does not predict.

Measuring the baseline first also says whether the heads are worth training at all. If
deterministic reconstruction already matches the corpus almost everywhere, there is
little for a head to add; the wider the gap, the more of the page is genuinely visual
information rather than arithmetic.

The rule implemented here is the common one: beams do not cross a beat, and a rest or a
note too long to carry a beam ends the group. Beat length comes from the metre - simple
metres beat on the denominator, compound metres on three of them. This is deliberately
the textbook rule rather than any particular engraver's house style, because that is what
"derivable without looking" means.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from homr.transformer.structured_notation import MAX_BEAM_LEVELS, BeamLevelState


@dataclass(frozen=True)
class BeamableNote:
    """What the rule needs to know about one note of a voice, in reading order."""

    #: Onset from the start of the measure, in MusicXML divisions.
    onset: int
    #: Sounding length in divisions.
    duration: int
    #: How many flags the written duration carries; 0 for a quarter or longer.
    flags: int
    is_rest: bool = False


def beat_divisions(beats: int, beat_type: int, divisions_per_quarter: int) -> int:
    """Length of one beaming unit in divisions.

    Compound metres - 6/8, 9/8, 12/8 - beat in threes, so their beaming unit is a dotted
    quarter rather than an eighth. Everything else beats on the denominator. 3/8 is
    treated as simple: it is one group of three, not a compound beat.
    """
    quarter = divisions_per_quarter
    unit = quarter * 4 // beat_type if beat_type else quarter
    if beat_type >= 8 and beats % 3 == 0 and beats > 3:
        return unit * 3
    return unit


def automatic_beams(notes: Sequence[BeamableNote], beat: int) -> list[tuple[BeamLevelState, ...]]:
    """Beam vectors the rule produces for one voice of one measure.

    Groups run while consecutive notes carry flags and stay inside one beat. A group of
    one is a flag, not a beam - a lone eighth on its own beat is drawn with a flag, and
    calling that a one-note beam would count a non-decision as a decision.
    """
    vectors: list[list[BeamLevelState]] = [
        [BeamLevelState.NOT_APPLICABLE] * MAX_BEAM_LEVELS for _ in notes
    ]
    for start, end in _groups(notes, beat):
        _beam_group(notes, vectors, start, end)
    return [tuple(vector) for vector in vectors]


def _groups(notes: Sequence[BeamableNote], beat: int) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive notes the rule beams together."""
    groups: list[tuple[int, int]] = []
    start: int | None = None
    for index, note in enumerate(notes):
        beamable = note.flags > 0 and not note.is_rest
        if not beamable:
            if start is not None:
                groups.append((start, index))
                start = None
            continue
        if start is None:
            start = index
        elif beat > 0 and note.onset // beat != notes[start].onset // beat:
            # A new beat starts a new group, which is the whole of the rule.
            groups.append((start, index))
            start = index
    if start is not None:
        groups.append((start, len(notes)))
    return groups


def _beam_group(
    notes: Sequence[BeamableNote],
    vectors: list[list[BeamLevelState]],
    start: int,
    end: int,
) -> None:
    for level in range(1, MAX_BEAM_LEVELS + 1):
        run_start: int | None = None
        for index in range(start, end + 1):
            carries = index < end and notes[index].flags >= level
            if carries and run_start is None:
                run_start = index
            elif not carries and run_start is not None:
                _write_run(vectors, run_start, index, level)
                run_start = None


def _write_run(vectors: list[list[BeamLevelState]], start: int, end: int, level: int) -> None:
    if end - start == 1:
        # One note at this level is a flag at this level, not a beam of length one.
        vectors[start][level - 1] = BeamLevelState.FLAG
        return
    vectors[start][level - 1] = BeamLevelState.BEGIN
    for index in range(start + 1, end - 1):
        vectors[index][level - 1] = BeamLevelState.CONTINUE
    vectors[end - 1][level - 1] = BeamLevelState.END


def agreement(
    predicted: Sequence[Sequence[BeamLevelState]],
    reference: Sequence[Sequence[BeamLevelState]],
    levels: int = MAX_BEAM_LEVELS,
) -> tuple[int, int]:
    """(matching notes, comparable notes), comparing whole beam vectors.

    A note counts as comparable when either side says any level applies to it, so notes
    that neither side beams - the overwhelming majority of a score - do not inflate the
    figure.
    """
    matching = comparable = 0
    for left, right in zip(predicted, reference, strict=True):
        first = tuple(left[:levels])
        second = tuple(right[:levels])
        if all(state == BeamLevelState.NOT_APPLICABLE for state in first + second):
            continue
        comparable += 1
        matching += first == second
    return matching, comparable
