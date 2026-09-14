"""Audit the corpus's own slur and tie labels against the constraints we enforce on ours.

Prompted by `IMSLP183800-sys5-v1`, a page carrying 7 slurs and 3 ties whose ground truth
records 2-3 slurs and 1 tie - and that one tie is a `stop` with no `start`.

This needs no source of truth beyond the labels themselves, which is what makes it worth
running. Three checks, each of which the reference must pass on its own terms:

  **Tie pairing.** A tie joins two notations of one pitch. A `start` whose pitch does not
  recur in the next chord, or a `stop` with no such `start` before it, is not a tie the
  corpus recorded - it is a label that cannot mean anything. Endpoints at the edge of a
  crop are excluded: a tie continuing into the next system is ordinary engraving.

  **Slur pairing.** Same, per slot: a `stop` in a slot nothing opened, or a `start` no
  slot closes, away from the crop edges.

  **Tokens against sidecar.** Both were written from one source in one pass, so a
  disagreement is our own converter contradicting itself. The comparison has to account
  for the collapse: the vocabulary maps `<tied>` and `<slur>` alike to
  `slurStart`/`slurStop` and then dedupes, so a note carrying a tie start *and* a slur
  start is one token and two sidecar endpoints **by design**. What is compared is
  therefore the set of endpoint kinds per note, not the count of endpoints.

A reference that fails these caps every number measured against it, and - worse - teaches
a head to omit what it omits.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from pathlib import Path

OPENS = {"start", "start_and_stop"}
CLOSES = {"stop", "start_and_stop"}


def _voices(sidecar_records: list[dict]) -> list[str]:
    """Each record's voice, or "unknown" for a sidecar written before voices existed."""
    return [str(record.get("voice", "unknown")) for record in sidecar_records]


def _onsets(sidecar_records: list[dict]) -> list[int | None]:
    """Each record's per-voice simultaneity index, or None before schema v6.

    Recording the voice alone did not recover adjacency: a token line is a simultaneity
    across every voice, so a voice's successive notes may share a line or sit several
    lines apart. This index is the relation a tie actually needs.
    """
    return [record.get("onsetIndex") for record in sidecar_records]


def _line(voices: list[str], notes: list, a: int, b: int) -> bool:
    """Whether two notes are on the same staff and, when both say, the same voice."""
    if notes[a][3] != notes[b][3]:
        return False
    if "unknown" in (voices[a], voices[b]):
        return True
    return voices[a] == voices[b]


def _notes(tokens_path: Path) -> list[tuple[str, bool, int, str, str]]:
    """(pitch, is_rest, chord id, staff, lift) per note-bearing entry, chords expanded.

    The staff is not optional on a grand staff. A tie start on the upper staff whose next
    chord belongs to the lower one has no partner *there*, and an audit that ignores the
    staff reports the corpus as broken when it is only two-handed.

    The lift - the written accidental - is carried because letter and octave alone do not
    identify a sounding pitch: E4 and E-flat4 are different notes and a tie cannot join
    them. It is compared leniently (see `_same_pitch`), because a tied note does not
    restate its neighbour's accidental.
    """
    out: list[tuple[str, bool, int, str]] = []
    chord = 0
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chord += 1
        for entry in line.split("&"):
            if "tieSlur" in entry:
                continue
            parts = entry.strip().split()
            if len(parts) < 2 or not parts[0].startswith(("note", "rest")):
                continue
            staff = parts[5] if len(parts) >= 6 else "upper"
            lift = parts[2] if len(parts) >= 3 else "_"
            out.append((parts[1], parts[0].startswith("rest"), chord, staff, lift))
    return out


#: Lifts that state nothing, so they cannot contradict a neighbour's accidental.
_UNSTATED_LIFTS = {"_", ".", "", "null"}


def _same_pitch(notes: list, a: int, b: int) -> bool:
    """Whether two entries are the same sounding pitch, as far as the tokens can say.

    Letter and octave must match. The written accidental is then required not to
    *contradict*: equal, or unstated on either side. A tie's second note does not restate
    the accidental, so demanding equality would call correct labels impossible; ignoring
    the lift entirely - which this audit used to do - lets E-flat4 pair with E-natural4.
    """
    if notes[a][0] != notes[b][0]:
        return False
    first, second = notes[a][4], notes[b][4]
    if first in _UNSTATED_LIFTS or second in _UNSTATED_LIFTS:
        return True
    return first == second


def _token_slur_endpoints(tokens_path: Path) -> int:
    total = 0
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        for entry in line.split("&"):
            parts = entry.strip().split()
            if len(parts) >= 5:
                total += sum(part in {"slurStart", "slurStop"} for part in parts[4].split("_"))
    return total


def _strictly_before(
    onsets: list[int | None],
    notes: list[tuple[str, bool, int, str]],
    earlier: int,
    later: int,
) -> bool:
    """Whether `earlier` sounds before `later`, by onset index when both carry one."""
    first, second = onsets[earlier], onsets[later]
    if first is not None and second is not None:
        return first < second
    return notes[earlier][2] < notes[later][2]


def _tie_partner_index(
    notes: list,
    index: int,
    voices: list[str] | None = None,
    onsets: list[int | None] | None = None,
) -> int | None:
    """The entry a tie at `index` would join, or None if a rest or the staff ends it.

    Searched within this note's own staff and voice: a tie joins one pitch on one line,
    and the next simultaneity in a flattened grand staff is as likely to be the other
    hand's. Among the members of that simultaneity, the one of the same pitch is returned
    when there is one - the tie's partner is a notehead, not a chord.
    """
    marks = voices or ["unknown"] * len(notes)
    indices = onsets or [None] * len(notes)
    chord = notes[index][2]

    def same(other: int) -> bool:
        return _line(marks, notes, index, other)

    def after(other: int) -> bool:
        here_index, there = indices[index], indices[other]
        if here_index is not None and there is not None:
            return there > here_index
        return notes[other][2] != chord

    def together(a: int, b: int) -> bool:
        first, second = indices[a], indices[b]
        if first is not None and second is not None:
            return first == second
        return notes[a][2] == notes[b][2]

    for later in range(index + 1, len(notes)):
        if not same(later) or not after(later):
            continue
        if notes[later][1]:
            return None
        members = [k for k in range(later, len(notes)) if together(k, later) and same(k)]
        for member in members:
            if _same_pitch(notes, index, member):
                return member
        return members[0] if members else None
    return None


def _tie_partner(
    notes: list,
    index: int,
    voices: list[str] | None = None,
    onsets: list[int | None] | None = None,
) -> str | None:
    """The pitch `_tie_partner_index` lands on, for callers that only compare pitches."""
    partner = _tie_partner_index(notes, index, voices, onsets)
    return None if partner is None else notes[partner][0]


def _first_of_its_line(notes: list, voices: list[str], index: int) -> bool:
    """Whether nothing on this note's own staff and voice sounds before it.

    A stop there has its start in the previous system. The old test asked only whether the
    note was the first of the entire crop, so a lower-staff tie continuing across a system
    break was reported as orphaned whenever any upper-staff note preceded it.
    """
    return not any(
        _line(voices, notes, other, index) and notes[other][2] < notes[index][2]
        for other in range(index)
    )


def _match_tie_endpoints(
    records: list[dict], notes: list, voices: list[str], onsets: list[int | None]
) -> set[int]:
    """Stops that a start of the same pitch actually reaches, each start used once.

    Adjacency is enforced through `_tie_partner`, which is the same next-simultaneity rule
    the starts are judged by - so a start and its stop agree about what "next" means
    instead of being scored by two different rules.
    """
    reached: set[int] = set()
    for index, record in enumerate(records):
        if record.get("tie", "none") not in OPENS:
            continue
        partner = _tie_partner_index(notes, index, voices, onsets)
        if partner is None or partner in reached:
            continue
        if records[partner].get("tie", "none") in CLOSES and _same_pitch(notes, index, partner):
            reached.add(partner)
    return reached


def audit(tokens_path: Path, sidecar_path: Path, counts: Counter) -> None:
    try:
        records = json.loads(sidecar_path.read_text(encoding="utf-8"))["notation"]
    except Exception:
        counts["unreadable"] += 1
        return
    notes = _notes(tokens_path)
    if len(notes) != len(records):
        counts["length_mismatch"] += 1
        return
    counts["files"] += 1
    counts["notes"] += len(notes)
    voices = _voices(records)
    onsets = _onsets(records)
    counts["voice_stated"] += sum(1 for v in voices if v != "unknown")

    # Which stops a start actually reaches. Walked per line and *consumed*, so one start
    # answers one stop: the old test let any earlier same-pitch start satisfy any later
    # stop, which counts a label joinable on the strength of a different tie entirely.
    matched_stops = _match_tie_endpoints(records, notes, voices, onsets)

    # Ties, against the constraint that defines them.
    for index, record in enumerate(records):
        state = record.get("tie", "none")
        if state == "none":
            continue
        counts["tie_endpoints"] += 1
        partner = _tie_partner_index(notes, index, voices, onsets)
        if state in OPENS:
            if partner is None:
                counts["tie_start_at_edge"] += 1
            elif _same_pitch(notes, index, partner):
                counts["tie_start_joinable"] += 1
            else:
                counts["tie_start_impossible"] += 1
        if state in CLOSES:
            if index in matched_stops:
                counts["tie_stop_joinable"] += 1
            elif _first_of_its_line(notes, voices, index):
                # Its partner is in the previous system, which this crop does not hold -
                # ordinary engraving, the same reasoning as a beam open at a crop edge.
                counts["tie_stop_at_edge"] += 1
            else:
                counts["tie_stop_orphan"] += 1

    # Slurs, per slot AND per line. `structured_notation_parser` allocates slots
    # independently for each source voice, so one global open-state per slot merged two
    # voices' spans into one: an upper-staff start could be closed by a lower-staff stop
    # and both were then counted as paired.
    width = max((len(r.get("slurs", [])) for r in records), default=0)
    open_in: dict[tuple[str, str, int], int] = {}
    for slot in range(width):
        for index, record in enumerate(records):
            slurs = record.get("slurs", [])
            if slot >= len(slurs):
                continue
            event = slurs[slot][0]
            if event == "none":
                continue
            counts["slur_endpoints"] += 1
            key = (notes[index][3], voices[index], slot)
            if event in CLOSES:
                if key in open_in:
                    counts["slur_paired"] += 1
                    del open_in[key]
                elif _first_of_its_line(notes, voices, index):
                    counts["slur_stop_at_edge"] += 1
                else:
                    counts["slur_stop_orphan"] += 1
            if event in OPENS:
                open_in[key] = index
    for key, index in open_in.items():
        last = max(
            (other for other in range(len(notes)) if _line(voices, notes, other, index)),
            default=index,
        )
        counts["slur_start_at_edge" if index == last else "slur_start_unclosed"] += 1

    # The token/sidecar endpoint comparison that used to sit here has been removed.
    # It could not mean what it was read to mean: the flat slur field is hoisted and
    # deduplicated per staff and simultaneity, while the sidecar records an endpoint per
    # notehead. The two are not equivalent representations by design, so a difference is
    # expected rather than diagnostic - and the figures it produced (18.3%, then 10.7%
    # after one correction) were quoted in this project as converter self-contradiction
    # when they were mostly that design. A permutation of the sidecar leaves them
    # untouched, which is what finally showed the check was measuring neither ordering
    # nor agreement.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.corpus.glob("*.notation.json"))
    if args.limit:
        paths = paths[: args.limit]
    counts: Counter[str] = Counter()
    for sidecar in paths:
        audit(Path(str(sidecar)[: -len(".notation.json")]), sidecar, counts)

    files, notes = counts["files"], counts["notes"]
    print(f"\n{args.corpus.name}: {files:,} files, {notes:,} notes")
    if counts["length_mismatch"] or counts["unreadable"]:
        print(
            f"  ({counts['length_mismatch']:,} length mismatch, {counts['unreadable']:,} unreadable)"
        )

    ties = counts["tie_endpoints"]
    print(f"\n  tie endpoints: {ties:,}")
    for key, label in (
        ("tie_start_joinable", "starts joining a repeat of the pitch"),
        ("tie_start_impossible", "starts whose pitch does NOT recur"),
        ("tie_stop_joinable", "stops with a start of that pitch before"),
        ("tie_stop_orphan", "stops with NO such start"),
        ("tie_start_at_edge", "starts continuing past the crop"),
        ("tie_stop_at_edge", "stops continuing past the crop"),
    ):
        if counts[key]:
            print(f"    {label:<42}{counts[key]:>8,}  ({counts[key]/max(ties,1):.1%})")

    slurs = counts["slur_endpoints"]
    print(f"\n  slur endpoints: {slurs:,}")
    for key, label in (
        ("slur_paired", "paired within the crop"),
        ("slur_stop_orphan", "stops nothing opened"),
        ("slur_start_unclosed", "starts nothing closes"),
        ("slur_stop_at_edge", "stops at the first note (edge)"),
        ("slur_start_at_edge", "starts at the last note (edge)"),
    ):
        if counts[key]:
            print(f"    {label:<42}{counts[key]:>8,}  ({counts[key]/max(slurs,1):.1%})")

    print(
        "\n  NOTE: this audit checks only whether a label can pair WITHIN its crop. It"
        "\n  cannot tell a wrong label from a label on a note the crop does not contain."
        "\n  For agreement with the engraved source, use source_tie_label_audit.py."
    )


if __name__ == "__main__":
    main()
