"""
Score-profile conditioning, layout half (design §7.2): turn a supplied `ScoreProfile`
and `system_grouping`'s geometric read of a page into a proposed staff-to-part mapping.

Deliberately built on top of `system_grouping.assign_voice_slots` rather than solving
staff identity from scratch. That function already answers the hard geometric question -
which voice slot (0..staves_per_system-1) a detected staff belongs to, including the
missing-staff case a real page's bracket detector gets wrong - using only staff spacing,
no knowledge of instruments at all. What it cannot do, because it has no profile to read,
is say voice slot 0 is "Violin I" rather than just "the first staff of the system". This
module is exactly that remaining step: profile parts, expanded to one `stableId` per
physical staff they occupy, laid directly against the voice-slot pattern.

This is a scored hypothesis, never an assertion - §7.1 says so explicitly, and the reason
is concrete: a system whose staff count does not match the profile is common (an
incomplete system, a divisi, a profile that is simply wrong about the piece) and must be
reported as a deviation the caller can see, not silently forced into a mapping that
would attach the wrong part's context to a staff's decoding.
"""

from dataclasses import dataclass

from homr.score_profile import ScorePart, ScoreProfile
from homr.system_grouping import SystemPartition


@dataclass(frozen=True)
class SystemPartAssignment:
    """One system's staff rows mapped onto profile parts, or the reasons it could not be."""

    #: Physical staff index (into the page's full staff list, matching
    #: `SystemPartition.groups`) -> the profile part's `stableId` it is proposed to be.
    #: Empty when nothing could be proposed for this system.
    staff_to_part: dict[int, str]
    deviations: tuple[str, ...]
    #: 1.0 when the system's detected staff count matches the profile's expected pattern
    #: exactly and every staff maps to a part; 0.0 when nothing could be proposed. There
    #: is no partial-credit score yet - see the module docstring on why silence is safer
    #: than a confident partial mapping until this has real evidence behind it.
    evidence_score: float


def propose_part_assignment(
    profile: ScoreProfile,
    partition: SystemPartition,
    voice_slots: list[tuple[int, ...] | None],
) -> list[SystemPartAssignment]:
    """One assignment per system in `partition.groups`, in the same order.

    `voice_slots` is `assign_voice_slots(staffs, partition)`'s own return value - passed
    in rather than recomputed so a caller that already has it (or a caller testing this
    function against a hand-built voice-slot pattern) never pays for or has to fake the
    geometry twice.
    """
    expected_pattern = profile.expected_staff_pattern
    assignments = []
    for group, slots in zip(partition.groups, voice_slots, strict=True):
        if slots is None:
            assignments.append(
                SystemPartAssignment(
                    staff_to_part={},
                    deviations=("voice slots could not be resolved geometrically",),
                    evidence_score=0.0,
                )
            )
            continue
        if len(expected_pattern) != partition.staves_per_system:
            assignments.append(
                SystemPartAssignment(
                    staff_to_part={},
                    deviations=(
                        f"profile expects {len(expected_pattern)} staff(s) per system, "
                        f"detected geometry implies {partition.staves_per_system}",
                    ),
                    evidence_score=0.0,
                )
            )
            continue
        staff_to_part = {
            staff_index: expected_pattern[slot]
            for staff_index, slot in zip(group, slots, strict=True)
        }
        assignments.append(
            SystemPartAssignment(
                staff_to_part=staff_to_part, deviations=(), evidence_score=1.0
            )
        )
    return assignments


def part_for_staff(
    assignments: list[SystemPartAssignment], system_index: int, staff_index: int
) -> str | None:
    """The proposed part for one staff, or None if this system's mapping has no opinion -
    an absent profile, an unresolved system, or a staff count that did not match are all
    the same "no opinion" to a caller, which is why this returns one type rather than
    exposing the deviation reasons again (`assignments[system_index].deviations` already
    carries those, for a caller that wants to show why)."""
    if not 0 <= system_index < len(assignments):
        return None
    return assignments[system_index].staff_to_part.get(staff_index)


def staff_to_part_by_system(
    profile: ScoreProfile, voice_present_by_system: list[list[bool]]
) -> list[dict[int, ScorePart]]:
    """The `staff_to_part` argument `cross_staff_consistency.findings_by_page`'s clef
    check needs, built the same way `staff_parsing.parse_staffs` itself is organised -
    by voice number, not by `SystemPartition` staff index.

    Voice number `v` is expected to be physical staff `v` of the profile's
    `expected_staff_pattern`, regardless of whether an earlier voice is absent from this
    particular system - a part's identity does not move just because a neighbour's
    staff went undetected there. What does move is *position*:
    `cross_staff_consistency.analyze_system` numbers a system's staves by which present
    voices it actually received, in order, not by voice number - so a system missing
    voice 0 must map its first staff (voice 1) to position 0, not to a `1` that has no
    meaning inside that system's own staff list. This mirrors
    `findings_by_page`'s own per-voice cursor exactly, so the two stay consistent by
    construction rather than by coincidence.

    A page whose profile does not even claim the same number of physical staves as were
    detected (`profile.total_staff_count != len(voice_present_by_system[0])`) gets an
    empty mapping for every system - the same "report nothing rather than guess"
    discipline `propose_part_assignment` already uses, for the same reason: a caller
    checking a decoded clef against the wrong instrument entirely is worse than a
    caller checking nothing.
    """
    pattern = profile.expected_staff_pattern
    voices = len(voice_present_by_system[0]) if voice_present_by_system else 0
    if len(pattern) != voices:
        return [{} for _ in voice_present_by_system]

    results = []
    for presence in voice_present_by_system:
        mapping: dict[int, ScorePart] = {}
        position = 0
        for voice_number, present in enumerate(presence):
            if not present:
                continue
            part = profile.part_by_id(pattern[voice_number])
            if part is not None:
                mapping[position] = part
            position += 1
        results.append(mapping)
    return results
