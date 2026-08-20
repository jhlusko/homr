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

Covers five of §12.1's eight listed findings so far: differing decoded measure counts,
conflicting key/time signatures, a clef inconsistent with a supplied score-profile part,
a slur left open or closed with nothing to match within one staff's own decode, and (a
later addition, from a design discussion rather than §12.1's original list) a shared
melodic/rhythmic motif across two staves that disagrees on one note's articulation. Not
yet covered, and not attempted here: conflicting barline *locations* (needs relative
position, not just count), a voice's measure duration disagreeing with the time
signature (needs duration arithmetic across the whole measure), part order changing
between systems and missing/extra staff output (both need state carried across systems,
not just within one) - left as further Stage A work, not silently assumed solved.

`analyze_system`'s input shape (one list of symbols per staff *of one system*) is not
`staff_parsing.parse_staffs`' output shape (one list per *voice*, concatenated across
every system that voice appears in). `split_by_system`/`findings_by_page` bridge the
two - see `findings_by_page`'s docstring for why that is a real reshaping problem, not
just a transpose, whenever a voice is missing from a system.
"""

import difflib
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


def _note_indices(symbols: list[EncodedSymbol]) -> list[int]:
    return [index for index, symbol in enumerate(symbols) if symbol.rhythm.startswith("note")]


def _shared_motif_findings(
    staff_a: list[EncodedSymbol],
    staff_b: list[EncodedSymbol],
    index_a: int,
    index_b: int,
    min_motif_length: int,
) -> list[Finding]:
    notes_a = _note_indices(staff_a)
    notes_b = _note_indices(staff_b)
    key_a = [(staff_a[k].rhythm, staff_a[k].pitch) for k in notes_a]
    key_b = [(staff_b[k].rhythm, staff_b[k].pitch) for k in notes_b]
    matcher = difflib.SequenceMatcher(a=key_a, b=key_b, autojunk=False)
    findings = []
    for block in matcher.get_matching_blocks():
        if block.size < min_motif_length:
            continue
        for offset in range(block.size):
            symbol_a = staff_a[notes_a[block.a + offset]]
            symbol_b = staff_b[notes_b[block.b + offset]]
            if symbol_a.articulation != symbol_b.articulation:
                findings.append(
                    Finding(
                        kind="motif_articulation_mismatch",
                        message=(
                            f"staves {index_a} and {index_b} play a matching "
                            f"{block.size}-note run but disagree on articulation at one "
                            f"note: {symbol_a.articulation!r} vs {symbol_b.articulation!r}"
                        ),
                        staff_indices=(index_a, index_b),
                    )
                )
    return findings


def check_shared_motifs(
    staves: list[list[EncodedSymbol]], min_motif_length: int = 4
) -> list[Finding]:
    """A run of at least `min_motif_length` notes with identical rhythm and pitch across
    two staves, but a differing articulation at one note within that run.

    Motivating case, from a design discussion rather than §12.1's original list: a fugal
    subject, an imitative entry, or a doubled passage played identically by two parts,
    where one part's decoded articulation reads staccato/accent/marcato differently from
    the other's - the surrounding identical notes are strong evidence one of the two
    misread, not that the parts are genuinely playing different articulations. No other
    Stage A check looks at note-level content at all; every other check compares one
    page-wide value per staff (key, time signature, clef).

    Deliberately narrow, and the limitation is real, not an oversight: a match requires
    identical rhythm *and* identical absolute pitch, so a transposed imitative entry (a
    fugal answer played a fifth higher, for example) is invisible to this check. A fuller
    version would need pitch-interval normalization instead of absolute-pitch equality -
    named as the natural next step, not built here.
    """
    findings: list[Finding] = []
    for i in range(len(staves)):
        for j in range(i + 1, len(staves)):
            findings.extend(_shared_motif_findings(staves[i], staves[j], i, j, min_motif_length))
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
    findings.extend(check_shared_motifs(staves))
    return findings


def split_by_system(symbols: list[EncodedSymbol]) -> list[list[EncodedSymbol]]:
    """One voice's concatenated-across-systems symbol stream, split back into one chunk
    per system, at the literal `"newline"` marker `staff_parsing.parse_staffs` inserts
    after every staff it decodes - including the last one for a voice, which is why a
    single trailing empty chunk (not an internal one) is dropped rather than kept: it is
    the artefact of that unconditional trailing marker, not a real empty system.
    """
    chunks: list[list[EncodedSymbol]] = [[]]
    for symbol in symbols:
        if symbol.rhythm == "newline":
            chunks.append([])
        else:
            chunks[-1].append(symbol)
    if chunks and not chunks[-1]:
        chunks.pop()
    return chunks


def staves_by_system(
    voices: list[list[EncodedSymbol]], voice_present_by_system: list[list[bool]]
) -> list[list[list[EncodedSymbol]]]:
    """Reshape `parse_staffs`' own output (one list per voice, that voice's symbols
    concatenated across every system it appears in) into one staves-list per system -
    the input shape `analyze_system` (and any repair proposal built from the same
    staves, e.g. `cross_staff_repair.propose_repairs`) expects.

    Not just a transpose. A voice missing from one system - `system_grouping`'s whole
    reason for existing is that this is common - contributes *no* chunk for that system
    at all, not an empty one, so `split_by_system`'s chunks cannot be lined up against
    systems by position alone. `voice_present_by_system[s][v]` (the same information
    `SystemPlan.staff_for_voice` already holds, passed as plain booleans rather than
    importing `staff_parsing.py`'s `SystemPlan` - that module pulls in cv2 and the
    transformer stack, which this package deliberately stays free of so it can be
    tested without either) says whether voice `v` contributed a chunk to system `s`, in
    the same order `voices` was built: consuming chunks with a per-voice cursor that
    only advances on `True` reproduces the correspondence exactly.
    """
    split_voices = [split_by_system(voice) for voice in voices]
    cursors = [0] * len(voices)
    result: list[list[list[EncodedSymbol]]] = []
    for presence in voice_present_by_system:
        staves = []
        for voice_number, present in enumerate(presence):
            if present:
                staves.append(split_voices[voice_number][cursors[voice_number]])
                cursors[voice_number] += 1
        result.append(staves)
    return result


def findings_by_page(
    voices: list[list[EncodedSymbol]],
    voice_present_by_system: list[list[bool]],
    staff_to_part_by_system: list[dict[int, ScorePart]] | None = None,
) -> list[list[Finding]]:
    """Stage A findings for every system on a page - see `staves_by_system` for the
    reshaping this relies on."""
    return [
        analyze_system(
            staves, staff_to_part_by_system[system_index] if staff_to_part_by_system else None
        )
        for system_index, staves in enumerate(staves_by_system(voices, voice_present_by_system))
    ]
