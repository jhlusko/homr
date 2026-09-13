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


def _line(voices: list[str], notes: list, a: int, b: int) -> bool:
    """Whether two notes are on the same staff and, when both say, the same voice."""
    if notes[a][3] != notes[b][3]:
        return False
    if "unknown" in (voices[a], voices[b]):
        return True
    return voices[a] == voices[b]


def _notes(tokens_path: Path) -> list[tuple[str, bool, int, str]]:
    """(pitch, is_rest, chord id, staff) per note-bearing entry, chord members expanded.

    The staff is not optional on a grand staff. A tie start on the upper staff whose next
    chord belongs to the lower one has no partner *there*, and an audit that ignores the
    staff reports the corpus as broken when it is only two-handed.
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
            out.append((parts[1], parts[0].startswith("rest"), chord, staff))
    return out


def _token_slur_endpoints(tokens_path: Path) -> int:
    total = 0
    for line in tokens_path.read_text(encoding="utf-8").splitlines():
        for entry in line.split("&"):
            parts = entry.strip().split()
            if len(parts) >= 5:
                total += sum(part in {"slurStart", "slurStop"} for part in parts[4].split("_"))
    return total


def _tie_partner(
    notes: list[tuple[str, bool, int, str]], index: int, voices: list[str] | None = None
) -> str | None:
    """The pitch a tie at `index` would join, or None if the staff or a rest ends it.

    Searched within this note's own staff: a tie joins one pitch on one staff, and the
    next chord in a flattened grand staff is as likely to be the other hand's.
    """
    pitch, _rest, chord, staff = notes[index]
    marks = voices or ["unknown"] * len(notes)

    def same(other: int) -> bool:
        return _line(marks, notes, index, other)

    for later in range(index + 1, len(notes)):
        if not same(later) or notes[later][2] == chord:
            continue
        if notes[later][1]:
            return None
        members = [
            notes[k][0]
            for k in range(later, len(notes))
            if notes[k][2] == notes[later][2] and same(k)
        ]
        return pitch if pitch in members else (members[0] if members else None)
    return None


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
    counts["voice_stated"] += sum(1 for v in voices if v != "unknown")

    # Ties, against the constraint that defines them.
    for index, record in enumerate(records):
        state = record.get("tie", "none")
        if state == "none":
            continue
        counts["tie_endpoints"] += 1
        partner = _tie_partner(notes, index, voices)
        if state in OPENS:
            if partner is None:
                counts["tie_start_at_edge"] += 1
            elif partner == notes[index][0]:
                counts["tie_start_joinable"] += 1
            else:
                counts["tie_start_impossible"] += 1
        if state in CLOSES:
            before = [
                i
                for i in range(index)
                if records[i].get("tie", "none") in OPENS
                and notes[i][0] == notes[index][0]
                and _line(voices, notes, i, index)
                and notes[i][2] < notes[index][2]
            ]
            if not before:
                counts["tie_stop_orphan" if index else "tie_stop_at_edge"] += 1
            else:
                counts["tie_stop_joinable"] += 1

    # Slurs, per slot.
    width = max((len(r.get("slurs", [])) for r in records), default=0)
    for slot in range(width):
        open_at = None
        for index, record in enumerate(records):
            slurs = record.get("slurs", [])
            if slot >= len(slurs):
                continue
            event = slurs[slot][0]
            if event == "none":
                continue
            counts["slur_endpoints"] += 1
            if event in CLOSES:
                if open_at is None:
                    counts["slur_stop_orphan" if index else "slur_stop_at_edge"] += 1
                else:
                    counts["slur_paired"] += 1
                    open_at = None
            if event in OPENS:
                open_at = index
        if open_at is not None:
            counts[
                "slur_start_at_edge" if open_at == len(records) - 1 else "slur_start_unclosed"
            ] += 1

    # What the flat field WOULD hold, not how many endpoints exist. The converter maps
    # both `<tied>` and `<slur>` to "slur" + type and then dedupes
    # (`music_xml_parser`: `slurs = list(set(slurs))`), because the six-branch vocabulary
    # collapses ties and slurs into one field and "slurStart_slurStart" is unrenderable.
    # So a note carrying a tie start and a slur start is one token and two sidecar
    # endpoints, by design. Comparing raw totals counted that as a disagreement.
    sidecar_total = 0
    for record in records:
        kinds = set()
        for event, _side in record.get("slurs", []):
            if event in OPENS:
                kinds.add("start")
            if event in CLOSES:
                kinds.add("stop")
        tie = record.get("tie", "none")
        if tie in OPENS:
            kinds.add("start")
        if tie in CLOSES:
            kinds.add("stop")
        sidecar_total += len(kinds)
    tokens_total = _token_slur_endpoints(tokens_path)
    counts["sidecar_endpoints"] += sidecar_total
    counts["token_endpoints"] += tokens_total
    if sidecar_total != tokens_total:
        counts["token_sidecar_disagree"] += 1


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
        f"\n  tokens vs sidecar: {counts['token_endpoints']:,} token endpoints, "
        f"{counts['sidecar_endpoints']:,} sidecar endpoints; "
        f"{counts['token_sidecar_disagree']:,} of {files:,} files disagree "
        f"({counts['token_sidecar_disagree']/max(files,1):.1%})"
    )


if __name__ == "__main__":
    main()
