"""Align scanned systems to score measures without assuming identical line breaks.

OpenScore preserves the printed line breaks it knows about, but MuseScore may add
extra breaks when it lays the score out.  A physical scan line can therefore cover
several consecutive reference systems (and, less commonly, the reverse).  Flat
``zip``-by-system-position is not a valid pairing operation.

This module aligns the two *whole-score* sequences by physical measure count.  It
supports many-to-many groups, skips false-positive scan systems and missing scan
systems, and only exposes an assignment when changing that assignment makes the
best global path measurably worse.  It never uses a recognizer prediction, so the
result is suitable for constructing an evaluation candidate set without leaking
either evaluated model into its labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import inf

MAX_GROUP = 4
GROUP_PENALTY = 0.25
COUNT_MISMATCH_PENALTY = 4.0
MAX_COUNT_MISMATCH = 2
SKIP_SYSTEM_PENALTY = 10.0
DEFAULT_MIN_MARGIN = 2.0


@dataclass(frozen=True)
class SystemAssignment:
    scan_index: int
    start_measure: int
    end_measure: int
    detected_measures: int
    source_system_start: int
    source_system_end: int
    count_exact: bool = True


@dataclass(frozen=True)
class AlignmentMove:
    kind: str
    scan_start: int
    scan_end: int
    source_start: int
    source_end: int
    cost: float
    assignments: tuple[SystemAssignment, ...] = ()


@dataclass(frozen=True)
class _Path:
    cost: float
    moves: tuple[AlignmentMove, ...]


def _prefix(values: list[int]) -> list[int]:
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result


def _move_assignments(
    scan_counts: list[int],
    source_prefix: list[int],
    i: int,
    a: int,
    j: int,
    b: int,
    *,
    exact: bool,
) -> tuple[SystemAssignment, ...]:
    # When grouped totals disagree, there is no defensible boundary between
    # several scan systems.  A single scan system can still consume the complete
    # source group as an uncertain anchor; it will be quarantined below, but keeps
    # a one-bar detector miss from shifting every later system.
    if not exact:
        if a != 1:
            return ()
        return (
            SystemAssignment(
                scan_index=i,
                start_measure=source_prefix[j],
                end_measure=source_prefix[j + b],
                detected_measures=scan_counts[i],
                source_system_start=j,
                source_system_end=j + b,
                count_exact=False,
            ),
        )
    cursor = source_prefix[j]
    result = []
    for scan_index in range(i, i + a):
        count = scan_counts[scan_index]
        result.append(
            SystemAssignment(
                scan_index=scan_index,
                start_measure=cursor,
                end_measure=cursor + count,
                detected_measures=count,
                source_system_start=j,
                source_system_end=j + b,
                count_exact=True,
            )
        )
        cursor += count
    return tuple(result)


def _best_path(
    scan_counts: list[int],
    source_counts: list[int],
    max_group: int,
    forbidden: tuple[int, int, int] | None = None,
) -> _Path:
    scan_prefix = _prefix(scan_counts)
    source_prefix = _prefix(source_counts)
    n, m = len(scan_counts), len(source_counts)

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> _Path:
        if i == n and j == m:
            return _Path(0.0, ())

        candidates: list[_Path] = []

        for a in range(1, min(max_group, n - i) + 1):
            if any(value <= 0 for value in scan_counts[i : i + a]):
                continue
            scan_total = scan_prefix[i + a] - scan_prefix[i]
            for b in range(1, min(max_group, m - j) + 1):
                source_total = source_prefix[j + b] - source_prefix[j]
                difference = abs(scan_total - source_total)
                if difference > MAX_COUNT_MISMATCH:
                    continue
                exact = difference == 0
                assignments = _move_assignments(
                    scan_counts, source_prefix, i, a, j, b, exact=exact
                )
                if forbidden is not None and any(
                    (item.scan_index, item.start_measure, item.end_measure) == forbidden
                    for item in assignments
                ):
                    continue
                tail = solve(i + a, j + b)
                move_cost = (
                    GROUP_PENALTY * ((a - 1) + (b - 1))
                    + COUNT_MISMATCH_PENALTY * difference
                )
                move = AlignmentMove(
                    "match", i, i + a, j, j + b, move_cost, assignments
                )
                candidates.append(_Path(move_cost + tail.cost, (move, *tail.moves)))

        if i < n:
            tail = solve(i + 1, j)
            move_cost = SKIP_SYSTEM_PENALTY + max(scan_counts[i], 1)
            move = AlignmentMove("skip_scan", i, i + 1, j, j, move_cost)
            candidates.append(_Path(move_cost + tail.cost, (move, *tail.moves)))

        if j < m:
            tail = solve(i, j + 1)
            move_cost = SKIP_SYSTEM_PENALTY + max(source_counts[j], 1)
            move = AlignmentMove("skip_source", i, i, j, j + 1, move_cost)
            candidates.append(_Path(move_cost + tail.cost, (move, *tail.moves)))

        # The secondary keys make output reproducible without weakening the
        # ambiguity check below: assignments shared by equally cheap paths remain
        # accepted, while assignments that differ receive a zero margin.
        return min(
            candidates,
            key=lambda path: (
                path.cost,
                sum(move.kind != "match" for move in path.moves),
                tuple((move.kind, move.scan_end, move.source_end) for move in path.moves),
            ),
        )

    return solve(0, 0)


def align_system_counts(
    scan_counts: list[int],
    source_counts: list[int],
    *,
    max_group: int = MAX_GROUP,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> dict:
    """Return a JSON-serialisable whole-score alignment report.

    A scan assignment is ``aligned`` only when the best path has an exact grouped
    measure-count match and the cheapest path that moves that scan elsewhere is at
    least ``min_margin`` worse.  Ambiguous and skipped scans remain in the report,
    but a corpus builder must not emit them.
    """
    if max_group < 1:
        raise ValueError("max_group must be at least 1")
    if any(value < 0 for value in (*scan_counts, *source_counts)):
        raise ValueError("system measure counts cannot be negative")

    best = _best_path(scan_counts, source_counts, max_group)
    chosen = {
        assignment.scan_index: assignment
        for move in best.moves
        for assignment in move.assignments
    }
    systems = []
    for scan_index, detected in enumerate(scan_counts):
        assignment = chosen.get(scan_index)
        if assignment is None:
            systems.append(
                {
                    "system": scan_index,
                    "detected_measures": detected,
                    "status": "skipped",
                    "reason": "no exact globally consistent measure-count match",
                }
            )
            continue

        alternative = _best_path(
            scan_counts,
            source_counts,
            max_group,
            forbidden=(
                assignment.scan_index,
                assignment.start_measure,
                assignment.end_measure,
            ),
        )
        margin = alternative.cost - best.cost if alternative.cost < inf else inf
        status = (
            "aligned"
            if assignment.count_exact and margin >= min_margin
            else "count_mismatch"
            if not assignment.count_exact
            else "ambiguous"
        )
        systems.append(
            {
                **asdict(assignment),
                "system": assignment.scan_index,
                "status": status,
                "margin": margin,
                "reason": (
                    None
                    if status == "aligned"
                    else "physical and source measure counts disagree"
                    if status == "count_mismatch"
                    else "equally plausible global alignment"
                ),
            }
        )

    return {
        "scan_counts": scan_counts,
        "source_counts": source_counts,
        "best_cost": best.cost,
        "max_group": max_group,
        "min_margin": min_margin,
        "systems": systems,
        "moves": [
            {
                **asdict(move),
                "assignments": [asdict(a) for a in move.assignments],
            }
            for move in best.moves
        ],
    }


def aligned_ranges(report: dict) -> dict[int, tuple[int, int]]:
    """The safe ``scan position -> [start, end)`` ranges from a report."""
    return {
        int(item["system"]): (int(item["start_measure"]), int(item["end_measure"]))
        for item in report.get("systems", [])
        if item.get("status") == "aligned"
    }
