"""Cross-staff findings as data, for a caller that has to show them to someone.

`_report_cross_staff_findings` has always written its findings with `eprint`. In a
deployment that means a container's stdout, which no reviewer will ever read - so a
profile that correctly caught a wrong clef, and a profile that was simply wrong about
the piece, produced exactly the same thing: nothing anyone sees.

This does not change what is logged. It adds an optional sink so an embedding caller can
collect the same findings structurally and put them in front of a person. Inactive by
default and cheap when inactive, because `homr` proper has no such caller and should not
pay for one.

The findings remain evidence, never corrections: nothing here decides anything, and a
caller that ignores the sink gets exactly today's behaviour.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_SINK: list[dict[str, Any]] | None = None


@contextmanager
def collect_findings() -> Iterator[list[dict[str, Any]]]:
    """Collect every finding recorded inside this block.

    Not re-entrant on purpose: two nested collectors would have to decide which one owns
    a finding, and there is no caller that needs both. The previous sink is restored on
    exit so a nested use degrades to the outer one rather than losing findings entirely.
    """
    global _SINK
    previous = _SINK
    collected: list[dict[str, Any]] = []
    _SINK = collected
    try:
        yield collected
    finally:
        _SINK = previous


def record(
    kind: str,
    message: str,
    *,
    system: int | None = None,
    staff_indices: tuple[int, ...] = (),
    part: str | None = None,
) -> None:
    """Record one finding, if anyone is collecting. A no-op otherwise.

    `staff_indices` are positions within the system, top to bottom, matching `Finding`'s
    own convention - never page-wide staff numbers, because these comparisons only ever
    happen within one system.
    """
    if _SINK is None:
        return
    finding: dict[str, Any] = {"kind": kind, "message": message}
    if system is not None:
        finding["system"] = system
    if staff_indices:
        finding["staffIndices"] = list(staff_indices)
    if part is not None:
        finding["part"] = part
    _SINK.append(finding)
