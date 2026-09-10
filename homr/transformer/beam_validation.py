"""
Whether a sequence of predicted beam states forms groups that could actually be engraved.

Per-token heads predict each note's beam vector independently, so nothing stops them
producing a locally plausible sequence that is globally impossible: a group that begins
and never ends, an END with no BEGIN before it, a level that continues across a note whose
duration cannot carry it. Those are not close calls - no engraver could draw them - and a
model that emits them is telling us something a per-token accuracy figure hides.

What this deliberately does not do is fix them. A validator that silently rewrote its
input would destroy the evidence: the raw prediction and any repair have to stay
distinguishable, or there is no way to tell a model that got the beaming right from one
whose output was corrected on the way out.

Validation runs per level and per voice, because that is the scope a beam group lives in.
Two voices on a staff beam independently, and level 2 can break where level 1 continues -
that secondary break is a real engraving choice, not an inconsistency.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from homr.transformer.structured_notation import BeamLevelState

#: States that continue an already-open group.
_INSIDE = {BeamLevelState.CONTINUE, BeamLevelState.END}


@dataclass
class BeamFindings:
    """Impossible constructions, counted and located rather than corrected."""

    #: A CONTINUE or END with no BEGIN before it.
    unopened: list[int] = field(default_factory=list)
    #: A BEGIN whose group never reaches an END.
    unclosed: list[int] = field(default_factory=list)
    #: A BEGIN inside a group that is already open.
    nested: list[int] = field(default_factory=list)
    #: A hook at level 1. A hook is a fragment of a secondary beam pointing back at a
    #: primary one; at level 1 there is no primary beam for it to attach to.
    hook_at_primary_level: list[int] = field(default_factory=list)
    #: A group of one note - a BEGIN immediately followed by its END.
    single_note_groups: list[int] = field(default_factory=list)
    groups: int = 0

    @property
    def valid(self) -> bool:
        return not (
            self.unopened
            or self.unclosed
            or self.nested
            or self.hook_at_primary_level
            or self.single_note_groups
        )

    def describe(self) -> str:
        problems = [
            f"{len(self.unopened)} unopened" if self.unopened else "",
            f"{len(self.unclosed)} unclosed" if self.unclosed else "",
            f"{len(self.nested)} nested" if self.nested else "",
            (
                f"{len(self.hook_at_primary_level)} hooks at level 1"
                if self.hook_at_primary_level
                else ""
            ),
            f"{len(self.single_note_groups)} single-note groups" if self.single_note_groups else "",
        ]
        listed = ", ".join(p for p in problems if p)
        return f"{self.groups} groups" + (f"; {listed}" if listed else "; valid")

    def merge(self, other: "BeamFindings") -> None:
        self.unopened += other.unopened
        self.unclosed += other.unclosed
        self.nested += other.nested
        self.hook_at_primary_level += other.hook_at_primary_level
        self.single_note_groups += other.single_note_groups
        self.groups += other.groups


def validate_level(states: Sequence[BeamLevelState], level: int) -> BeamFindings:
    """Check one beam level's states across one voice, in reading order.

    Positions are indices into `states`, so a caller can point at the offending note
    rather than reporting that something, somewhere, was wrong.
    """
    findings = BeamFindings()
    open_at: int | None = None
    length_since_open = 0

    for index, state in enumerate(states):
        if state == BeamLevelState.BEGIN:
            if open_at is not None:
                findings.nested.append(index)
                findings.unclosed.append(open_at)
            open_at = index
            length_since_open = 1
        elif state in _INSIDE:
            if open_at is None:
                findings.unopened.append(index)
                continue
            length_since_open += 1
            if state == BeamLevelState.END:
                if length_since_open < 2:
                    findings.single_note_groups.append(open_at)
                findings.groups += 1
                open_at = None
        elif state in (BeamLevelState.FORWARD_HOOK, BeamLevelState.BACKWARD_HOOK):
            if level == 1:
                findings.hook_at_primary_level.append(index)
            # A hook sits inside a group at its own level without closing it, so an open
            # group is left open on purpose.
            if open_at is not None:
                length_since_open += 1
        elif open_at is not None:
            # FLAG or NOT_APPLICABLE interrupts a group that was never closed.
            findings.unclosed.append(open_at)
            open_at = None

    if open_at is not None:
        findings.unclosed.append(open_at)
    return findings


def validate_voice(vectors: Sequence[Sequence[BeamLevelState]], levels: int) -> BeamFindings:
    """Check every level of one voice's beam vectors."""
    findings = BeamFindings()
    for level in range(1, levels + 1):
        states = [
            vector[level - 1] if level - 1 < len(vector) else BeamLevelState.NOT_APPLICABLE
            for vector in vectors
        ]
        findings.merge(validate_level(states, level))
    return findings
