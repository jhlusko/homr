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

Covers seven of §12.1's eight listed findings so far: differing decoded measure counts,
a measure's total note/rest duration disagreeing with the rest of the system
(`check_measure_durations` - see its own docstring for why this compares content
duration rather than a decoded time-signature numerator, which does not exist: the
decoder only ever states a denominator), conflicting key/time signatures, a clef
inconsistent with a supplied score-profile part, a slur left open or closed with nothing
to match within one staff's own decode, missing or extra staff output relative to the
rest of a page (`check_page_staff_counts` - the one check here that is page-wide over
staff *counts* rather than one-system, since a count means nothing on its own without
the rest of the page to compare against), part order changing between systems
specifically (`check_page_staff_counts` catches a count changing; `check_part_order` -
also page-wide, needs a score profile - catches the parts that all stay present
nonetheless swapping position between one system and the next), and (a later addition,
from a design discussion rather than §12.1's original list) a shared melodic/rhythmic
motif across two staves that disagrees on one note's articulation. Not yet covered, and
not attempted here: conflicting barline *locations* (needs relative position, not just
count or total duration) - the one remaining named item from §12.1's original eight -
left as further Stage A work, not silently assumed solved.

`analyze_system`'s input shape (one list of symbols per staff *of one system*) is not
`staff_parsing.parse_staffs`' output shape (one list per *voice*, concatenated across
every system that voice appears in). `split_by_system`/`findings_by_page` bridge the
two - see `findings_by_page`'s docstring for why that is a real reshaping problem, not
just a transpose, whenever a voice is missing from a system.
"""

import difflib
import statistics
from dataclasses import dataclass
from fractions import Fraction

from homr.music_xml_generator import group_into_chords
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


def _measure_durations(staff: list[EncodedSymbol]) -> list[Fraction]:
    """Total note/rest duration within each measure of one staff, in whole-note units -
    the same per-measure accumulation `homr.music_xml_generator`'s
    `find_division_and_time_signature_nominator` already does to *infer* a time
    signature's numerator, since the decoder never states one: `build_rhythm` only ever
    emits `timeSignature/<denominator>` (see that module's docstring) - the numerator
    MusicXML generation writes is computed from measure content, not decoded. That makes
    a measure's content duration the only independently available "what does this staff
    think this measure adds up to" signal there is, which is what this compares across
    staves in `check_measure_durations` below.

    Reuses `group_into_chords` rather than reimplementing chord grouping here - getting
    that wrong would silently double-count a chord's simultaneous notes as if they were
    sequential, the same class of bug `staves_by_system` was built to avoid for its own
    reshaping problem.
    """
    duration_in_measure = Fraction(0)
    durations = []
    for chord in group_into_chords(staff):
        if chord.is_barline():
            if duration_in_measure > Fraction(0):
                durations.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        else:
            duration_in_measure += chord.get_duration()
    if duration_in_measure > Fraction(0):
        durations.append(duration_in_measure)
    return durations


def check_measure_durations(staves: list[list[EncodedSymbol]]) -> list[Finding]:
    """One staff's typical measure duration disagreeing with the rest of the system's -
    complements `check_measure_counts`: the same barline count can still hide a wrong
    total duration within a measure (a dropped or extra beat that does not change how
    many barlines were decoded).

    Compares each staff's *median* measure duration, not every measure pairwise - the
    same robustness `find_division_and_time_signature_nominator` relies on, so one
    truncated or unusually-notated measure does not by itself flag an otherwise
    consistent staff, and staves are free to differ measure-by-measure in real
    polyphonic music (a passage where one voice rests through what another plays)
    without disagreeing on the prevailing pulse. A staff with no measures at all (no
    barlines, or nothing but zero-duration content) contributes no median and is
    silently excluded rather than treated as a mismatch - nothing to compare it against.
    """
    medians: dict[int, Fraction] = {}
    for index, staff in enumerate(staves):
        durations = _measure_durations(staff)
        if durations:
            medians[index] = Fraction(statistics.median(durations))
    if len(set(medians.values())) <= 1:
        return []
    return [
        Finding(
            kind="measure_duration_mismatch",
            message=(
                "typical measure duration (whole notes) disagrees across the system: "
                f"{ {index: str(value) for index, value in medians.items()} }"
            ),
            staff_indices=tuple(sorted(medians)),
        )
    ]


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
    findings.extend(check_measure_durations(staves))
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


def check_page_staff_counts(voice_present_by_system: list[list[bool]]) -> list[Finding]:
    """A system whose staff count differs from the page's dominant (most common) staff
    count - one of §12.1's originally-named checks that stayed unbuilt the longest:
    missing or extra staff output. Genuinely page-wide, unlike every other check in this
    module: one system's staff count means nothing in isolation, only against what the
    rest of the page does, so this is the one check that needs every system at once
    rather than one system in isolation.

    A tie for "most common" count resolves toward the larger value - a system missing
    one voice (a dropped staff, `system_grouping`'s own reason for existing) is a far
    more common real failure than a system that gained a staff no other system has, so
    "most staves" is the safer tie-break default when the page gives no other signal.

    Deliberately does not reuse `Finding.staff_indices`' usual meaning (positions within
    one system's staves) - this compares *systems* against each other, not staves within
    one, so `staff_indices` is left empty here and which systems disagree is named in
    `message` instead, rather than silently overloading a field whose own docstring
    describes something different.
    """
    counts = [sum(1 for present in system if present) for system in voice_present_by_system]
    if len(set(counts)) <= 1:
        return []
    majority = max(sorted(set(counts), reverse=True), key=counts.count)
    return [
        Finding(
            kind="page_staff_count_mismatch",
            message=(
                f"system {system_index} has {count} staves, most of the page has "
                f"{majority}: per-system counts are {counts}"
            ),
            staff_indices=(),
        )
        for system_index, count in enumerate(counts)
        if count != majority
    ]


def _ordered_parts(part_map: dict[int, ScorePart]) -> list[str]:
    return [part_map[index].stable_id for index in sorted(part_map)]


def check_part_order(staff_to_part_by_system: list[dict[int, ScorePart]]) -> list[Finding]:
    """Two consecutive systems disagreeing on the relative order of the parts they both
    resolved a mapping for - the last of §12.1's originally-named checks: part order
    changing between systems. Needs a score profile (the same `staff_to_part_by_system`
    mapping `check_clefs_against_profile` and `check_page_staff_counts` use, the latter
    for a plain count rather than identity) to know which staff is which *part*, not
    just how many staves there are - a count changing is already `check_page_staff_
    counts`' territory; this catches the parts that all stay present nonetheless
    swapping position, which a count comparison cannot see at all.

    Compares each system only against the one immediately before it, restricted to the
    parts *both* systems resolved a mapping for - a part missing a mapping in either
    system (already covered elsewhere: `check_page_staff_counts` for a genuine staff-
    count change, or simply an unresolved profile match) is dropped from the comparison
    rather than treated as evidence of a swap. A restricted list of 0 or 1 parts is
    trivially "in order" and produces no finding - there is nothing to compare.
    """
    findings = []
    previous_order: list[str] | None = None
    previous_index = -1
    for system_index, part_map in enumerate(staff_to_part_by_system):
        order = _ordered_parts(part_map)
        if previous_order is not None:
            common_previous = [part for part in previous_order if part in order]
            common_current = [part for part in order if part in previous_order]
            if len(common_current) > 1 and common_previous != common_current:
                findings.append(
                    Finding(
                        kind="part_order_mismatch",
                        message=(
                            f"system {system_index} orders shared parts {common_current}, "
                            f"system {previous_index} ordered them {common_previous}"
                        ),
                        staff_indices=(),
                    )
                )
        previous_order = order
        previous_index = system_index
    return findings


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
