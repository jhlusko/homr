"""
Cross-staff context and repair, Stage A (design §12.1): deterministic consistency
analysis over a system's already-decoded staves.

Every staff in `homr`'s pipeline decodes independently - nothing about the transformer
sees another staff's output. That independence is why a system of four parts can still
disagree with itself: one viola measure decoded in 7/8 while the other three parts agree
on 4/4 is not a rare edge case, it is what an independently-decoded low-confidence token
looks like from the outside. Stage A finds these disagreements; it does not fix them
(Stage B, not built) and it never touches MusicXML - it only emits structured `Finding`s
a caller can act on, log, or hand to a human reviewer.

Operates on already-decoded `EncodedSymbol` sequences (`homr.transformer.vocabulary`'s
token stream), one list per staff of a system - not on images, not on `Staff` geometry -
the same deliberate decoupling `score_profile_layout.py` uses, and for the same reason:
this can be built and tested against hand-built token sequences without running the
transformer or touching a real page.

Covers four of §12.1's eight listed findings so far: differing decoded measure counts,
conflicting key/time signatures, a clef inconsistent with a supplied score-profile part,
and a slur left open or closed with nothing to match within one staff's own decode. Not
yet covered, and not attempted here: conflicting barline *locations* (needs relative
position, not just count), a voice's measure duration disagreeing with the time
signature (needs duration arithmetic across the whole measure), part order changing
between systems and missing/extra staff output (both need state carried across systems,
not just within one) - left as further Stage A work, not silently assumed solved.
"""

from dataclasses import dataclass

from homr.score_profile import ScorePart
from homr.transformer.structured_notation import SlurEvent
from homr.transformer.vocabulary import EncodedSymbol

_BARLINE_RHYTHMS = ("barline", "doublebarline", "bolddoublebarline")


@dataclass(frozen=True)
class Finding:
    """One deterministic disagreement, evidence rather than a correction.

    `staff_indices` are positions within the system (0-based, top to bottom) that the
    finding concerns - never a page-wide staff index, since Stage A only ever compares
    staves within one system.
    """

    kind: str
    message: str
    staff_indices: tuple[int, ...]


def measure_count(symbols: list[EncodedSymbol]) -> int:
    return sum(1 for symbol in symbols if symbol.rhythm in _BARLINE_RHYTHMS)


def _key_signature_sequence(symbols: list[EncodedSymbol]) -> tuple[str, ...]:
    """Every keySignature token in decoded order, not deduplicated against a running
    value - two staves that agree on the *first* key but disagree on when or whether it
    later changes are still a real inconsistency, and only the full sequence shows it."""
    return tuple(symbol.rhythm for symbol in symbols if symbol.rhythm.startswith("keySignature"))


def _time_signature_sequence(symbols: list[EncodedSymbol]) -> tuple[str, ...]:
    return tuple(symbol.rhythm for symbol in symbols if symbol.rhythm.startswith("timeSignature"))


def _first_clef(symbols: list[EncodedSymbol]) -> str | None:
    return next((symbol.rhythm for symbol in symbols if symbol.rhythm.startswith("clef")), None)


def check_measure_counts(staves: list[list[EncodedSymbol]]) -> list[Finding]:
    counts = [measure_count(staff) for staff in staves]
    if len(set(counts)) <= 1:
        return []
    return [
        Finding(
            kind="measure_count_mismatch",
            message=f"decoded measure counts disagree across the system: {counts}",
            staff_indices=tuple(range(len(staves))),
        )
    ]


def check_key_signatures(staves: list[list[EncodedSymbol]]) -> list[Finding]:
    sequences = [_key_signature_sequence(staff) for staff in staves]
    if len(set(sequences)) <= 1:
        return []
    return [
        Finding(
            kind="key_signature_mismatch",
            message=f"key signature sequences disagree across the system: {sequences}",
            staff_indices=tuple(range(len(staves))),
        )
    ]


def check_time_signatures(staves: list[list[EncodedSymbol]]) -> list[Finding]:
    sequences = [_time_signature_sequence(staff) for staff in staves]
    if len(set(sequences)) <= 1:
        return []
    return [
        Finding(
            kind="time_signature_mismatch",
            message=f"time signature sequences disagree across the system: {sequences}",
            staff_indices=tuple(range(len(staves))),
        )
    ]


def check_clefs_against_profile(
    staves: list[list[EncodedSymbol]], staff_to_part: dict[int, ScorePart]
) -> list[Finding]:
    """A decoded clef the supplied profile did not expect for that staff's part.

    Silent (no finding) for a staff with no proposed part, or a part with no stated
    `likelyClefs` - an empty expectation is "we do not know," per §7.1, never "nothing is
    valid here."
    """
    findings = []
    for index, staff in enumerate(staves):
        part = staff_to_part.get(index)
        if part is None or not part.likely_clefs:
            continue
        clef = _first_clef(staff)
        if clef is None:
            continue
        clef_name = clef.removeprefix("clef_")
        if clef_name not in part.likely_clefs:
            findings.append(
                Finding(
                    kind="clef_profile_mismatch",
                    message=(
                        f"staff {index} decoded clef {clef_name!r}, profile part "
                        f"{part.stable_id!r} expects one of {part.likely_clefs}"
                    ),
                    staff_indices=(index,),
                )
            )
    return findings


def check_dangling_slurs(staves: list[list[EncodedSymbol]]) -> list[Finding]:
    """A slur slot left open at the end of a staff's decode, or a STOP/START_AND_STOP
    with nothing open in that slot - both mean the decode itself is inconsistent with
    what a slur is (a span with two ends), independent of any other staff.

    Only staves whose symbols carry structured slur notation are checked (`symbol.notation`
    is None for a run that predates the structured heads, or a config that never asked for
    them) - absence of the data is not evidence of a dangling slur.
    """
    findings = []
    for index, staff in enumerate(staves):
        open_slots: set[int] = set()
        for symbol in staff:
            if symbol.notation is None:
                continue
            for slot, (event, _side) in enumerate(symbol.notation.slurs):
                if event == SlurEvent.START:
                    open_slots.add(slot)
                elif event == SlurEvent.STOP:
                    if slot not in open_slots:
                        findings.append(
                            Finding(
                                kind="dangling_slur_stop",
                                message=f"staff {index} slot {slot}: slur stop with no open start",
                                staff_indices=(index,),
                            )
                        )
                    open_slots.discard(slot)
                elif event == SlurEvent.START_AND_STOP:
                    if slot not in open_slots:
                        findings.append(
                            Finding(
                                kind="dangling_slur_stop",
                                message=(
                                    f"staff {index} slot {slot}: slur stop-and-start with "
                                    "no open start"
                                ),
                                staff_indices=(index,),
                            )
                        )
                    open_slots.add(slot)
        for slot in sorted(open_slots):
            findings.append(
                Finding(
                    kind="dangling_slur_start",
                    message=f"staff {index} slot {slot}: slur never closed",
                    staff_indices=(index,),
                )
            )
    return findings


def analyze_system(
    staves: list[list[EncodedSymbol]],
    staff_to_part: dict[int, ScorePart] | None = None,
) -> list[Finding]:
    """Every Stage A finding for one system, in a stable order.

    `staff_to_part` is optional - the clef-vs-profile check is the only one that needs
    it, and every other check runs regardless of whether a score profile was ever
    supplied.
    """
    findings = []
    findings.extend(check_measure_counts(staves))
    findings.extend(check_key_signatures(staves))
    findings.extend(check_time_signatures(staves))
    if staff_to_part:
        findings.extend(check_clefs_against_profile(staves, staff_to_part))
    findings.extend(check_dangling_slurs(staves))
    return findings
