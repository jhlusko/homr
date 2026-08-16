"""
Deterministic page-level grouping of physical staves into systems.

Staff detection yields an ordered list of physical staves down the page. Turning that
into systems is a partition problem, and homr's existing answer to it has a blind spot:
when the bracket/barline detector produces rows that disagree with each other, staff
parsing falls back to finding a repeating period in each staff's is_grandstaff flag.
For an ensemble of same-type single-staff parts - a string quartet is the canonical case
- that flag is constant down the whole page, so period 1 fits trivially and every staff
becomes its own system. The page then decodes as a single part containing every staff's
music in sequence.

This module decides the partition from page geometry instead. The signal it uses is the
one an engraver actually controls: staves inside a system sit closer together than
systems do. Measured in staff unit sizes, on a page where the bracket detector produced
the inconsistent rows [3, 4, 3, 1, 4, 4] for what is really five 4-staff systems, the
gaps separate cleanly:

    within a system   3.6 - 6.7 unit sizes
    between systems   8.7 - 9.1 unit sizes

A page of genuinely independent single staves has no such split - every gap is a system
gap - which is what makes this decidable rather than a guess, and why the result carries
a `confident` flag instead of always returning an answer.

Robustness matters more than sharpness here. That same page has one 14.8-unit gap in the
middle of its first system, because staff detection missed a staff there. Any rule of the
form "cut at the largest gaps" puts a system boundary in the wrong place. So the partition
is chosen page-wide, under the constraint that systems have a uniform staff count, and
scored on how well the cuts separate from the internals on average - a single outlying
gap cannot move a boundary on its own.
"""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from homr import constants
from homr.model import Staff
from homr.simple_logging import eprint


@dataclass(frozen=True)
class SystemPartition:
    """One candidate reading of the page: staves grouped into systems."""

    staves_per_system: int
    groups: tuple[tuple[int, ...], ...]
    #: Mean cut gap minus mean internal gap, in staff unit sizes. Higher is a cleaner
    #: separation between "next staff of this system" and "first staff of the next".
    separation: float
    #: Cuts that split a staff pair the bracket/barline detector had joined. Geometry and
    #: bracket evidence disagreeing is a reason to distrust the partition.
    broken_connections: int

    @property
    def score(self) -> float:
        return self.separation - constants.broken_connection_penalty * self.broken_connections

    def describe(self) -> str:
        sizes = [len(group) for group in self.groups]
        return (
            f"{len(self.groups)} systems of {sizes}, separation "
            f"{self.separation:.1f} unit sizes"
            + (f", {self.broken_connections} broken connections" if self.broken_connections else "")
        )


@dataclass(frozen=True)
class GroupingResult:
    best: SystemPartition
    runner_up: SystemPartition | None
    #: False when the page carries no usable evidence of multi-staff systems - a solo
    #: part, or too few staves to tell. Callers must keep their existing behaviour then
    #: rather than impose this partition.
    confident: bool


def _normalized_gaps(staffs: Sequence[Staff]) -> list[float]:
    """Vertical gap above each staff, in unit sizes; index i is the gap between i-1 and i.

    Normalising by unit size is what makes a threshold meaningful across scans of
    different resolutions, and it is per-pair rather than per-page so a score whose staff
    size changes down the page (an ossia, a reduced final system) stays comparable.

    Returns [] when two staffs overlap vertically, which means the staff list itself is
    not a sequence of distinct staffs down the page - the precondition this whole module
    rests on. It happens: one quartet page had a staff line detected twice, once as its
    left half and once as its right, overlapping by 4.4 unit sizes. Partitioning that
    list produces a system holding two halves of one staff, which then crops to a
    degenerate image. Deciding which detection to keep is a staff-detection question, so
    the honest answer here is to have no opinion and let the caller fall back.
    """
    gaps = []
    for index in range(1, len(staffs)):
        previous, current = staffs[index - 1], staffs[index]
        unit_size = (previous.average_unit_size + current.average_unit_size) / 2
        if unit_size <= 0:
            return []
        gap = (current.min_y - previous.max_y) / unit_size
        if gap < constants.min_gap_for_distinct_staffs:
            return []
        gaps.append(gap)
    return gaps


def _candidate_groupings(count: int, staves_per_system: int) -> list[tuple[tuple[int, ...], ...]]:
    """Ways to cut `count` staves into systems of `staves_per_system`.

    Exactly one system may come up short, and it may sit anywhere on the page. A short
    system at an edge is the page's own incomplete first or last one; a short system in
    the middle is a complete one that staff detection came up a staff on. Both are
    common - on one quartet score the bracket rows read [4, 4, 3, 4, 4] and [4, 3, 4, 4,
    4] on consecutive pages - and restricting the short system to the edges would leave
    the whole page unreadable for the sake of one missing staff.
    """
    size = staves_per_system
    if size < 1 or count < size:
        return []
    full, remainder = divmod(count, size)
    if full < constants.min_systems_for_geometric_grouping:
        return []

    def tile(short_at: int | None) -> tuple[tuple[int, ...], ...]:
        groups: list[tuple[int, ...]] = []
        start = 0
        for position in range(full + 1):
            length = remainder if position == short_at else size
            if length == 0 or start >= count:
                continue
            groups.append(tuple(range(start, start + length)))
            start += length
        return tuple(groups)

    if remainder == 0:
        return [tile(None)]
    return [tile(position) for position in range(full + 1)]


def _evaluate(
    groups: tuple[tuple[int, ...], ...],
    gaps: list[float],
    connected_pairs: set[tuple[int, int]],
    staves_per_system: int,
) -> SystemPartition | None:
    cut_indices = {group[0] for group in groups[1:]}
    cuts = [gaps[index - 1] for index in cut_indices]
    internals = [gaps[index - 1] for index in range(1, len(gaps) + 1) if index not in cut_indices]
    if not cuts or not internals:
        return None
    broken = sum(1 for index in cut_indices if (index - 1, index) in connected_pairs)
    separation = statistics.mean(cuts) - statistics.mean(internals)
    # A single wide internal gap (a missed staff) drags the internal mean up, so the mean
    # alone can stay positive for a partition whose boundaries are in the wrong place.
    # Requiring every cut to clear the typical internal gap by a margin is the check that
    # actually pins the boundaries, and using the median keeps that outlier from setting
    # the bar.
    if min(cuts) <= statistics.median(internals) * constants.min_cut_to_internal_gap_ratio:
        return None
    return SystemPartition(staves_per_system, groups, separation, broken)


def find_system_grouping(
    staffs: Sequence[Staff], connected_pairs: set[tuple[int, int]]
) -> GroupingResult | None:
    """Partition ordered staves into systems using page geometry.

    connected_pairs holds adjacent staff indices the bracket/barline detector placed in
    the same row. It is corroborating evidence, not a constraint: that detector is what
    produced the inconsistent rows this module exists to repair, so a partition is
    penalised for contradicting it but never forced to obey it.

    Returns None when there is nothing to decide (fewer than two staves), otherwise a
    result whose `confident` flag says whether the caller should act on it.
    """
    if len(staffs) < 2:
        return None
    gaps = _normalized_gaps(staffs)
    if not gaps:
        return None

    partitions: list[SystemPartition] = []
    max_size = min(
        len(staffs) // constants.min_systems_for_geometric_grouping, constants.max_staves_per_system
    )
    for staves_per_system in range(2, max_size + 1):
        for groups in _candidate_groupings(len(staffs), staves_per_system):
            partition = _evaluate(groups, gaps, connected_pairs, staves_per_system)
            if partition is not None:
                partitions.append(partition)
    if not partitions:
        return None

    partitions.sort(key=lambda p: p.score, reverse=True)
    best = partitions[0]
    runner_up = partitions[1] if len(partitions) > 1 else None
    confident = (
        best.separation >= constants.min_system_gap_separation
        and best.broken_connections <= constants.max_broken_connections_for_grouping
    )
    return GroupingResult(best, runner_up, confident)


def report_grouping(result: GroupingResult) -> None:
    """Surface the decision and its closest competitor, so an ambiguous page is visible
    rather than silently resolved."""
    verdict = "using" if result.confident else "rejected (weak evidence)"
    eprint(f"Page geometry suggests {result.best.describe()} - {verdict}")
    if result.runner_up is not None:
        eprint(f"  next best was {result.runner_up.describe()}")
