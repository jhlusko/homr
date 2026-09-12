"""The beam discrepancy inventory, measured on scans instead of symbolic files.

`beam_discrepancy_inventory` walks OpenScore MusicXML, which is clean and has no image.
This walks the corrected scanned corpus, where every staff crop carries three things that
were joined when the corpus was built:

    sq7541288_0005_0001_1.png                 the scan
    sq7541288_0005_0001_1.txt                 clef, metre, durations, barlines
    sq7541288_0005_0001_1.txt.notation.json   the engraved beams

Two reasons this is the better measurement. Every case comes with **its own crop**, with
no join to get wrong - and this project's worst defect came from joining scans to labels
across editions whose pagination differed, where every guard passed and the pairs were
still wrong. And it measures the **scanned** domain: the beam head scores 90.0% on
synthetic and 69.1% on scans, so a clean-symbolic inventory describes a domain the
scanner never sees.

**The sidecar is one entry per note, chord members counted separately.** Chords are
`&`-joined on one token line, so a line with one `&` contributes two entries. Checked on
400 random crops before being relied on: 400 of 400 agree. A crop whose counts disagree is
skipped and counted rather than truncated into alignment.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.structured_notation import BeamLevelState
from training.omr_datasets.unbeamed_review_set import is_beamed

#: Divisions per quarter note for the synthetic onset grid. The token stream gives
#: durations as fractions of a whole note, so this only has to be fine enough that every
#: duration lands on an integer - 64ths with a dot need 48.
DIVISIONS = 48

#: `note_<denominator>`, optionally dotted: the denominator says how many of this note
#: fit in a whole note, so it is not restricted to powers of two - `note_12` is a triplet
#: eighth. That distinction matters twice over, because the *sounded* duration and the
#: *written* value diverge for tuplets: a triplet eighth lasts a third of a quarter and is
#: still drawn with one flag. Onsets need the first, beaming needs the second.


def token_duration(token: str) -> tuple[Fraction, int] | None:
    """(duration in quarter notes, flag count), or None if not a note or rest."""
    if not token.startswith(("note_", "rest_")):
        return None
    body = token.split("_", 1)[1]
    digits = ""
    for character in body:
        if character.isdigit():
            digits += character
        else:
            break
    if not digits:
        return None
    denominator = int(digits)
    if denominator <= 0:
        return None
    duration = Fraction(4, denominator)
    # The written value is the largest power of two that fits, which is what a triplet
    # eighth (12) and an ordinary eighth (8) have in common: both are drawn as eighths.
    written = 1
    while written * 2 <= denominator:
        written *= 2
    flags = 0
    value = written
    while value >= 8:
        flags += 1
        value //= 2
    for character in body[len(digits) :]:
        if character != ".":
            break
        duration += duration / 2
    return duration, flags


def parse_metre(lines: list[str]) -> tuple[int, int]:
    for line in lines:
        head = line.split()[0]
        if head.startswith("timeSignature"):
            body = head.split("timeSignature", 1)[1].lstrip("_")
            if "/" in body:
                beats, _, beat_type = body.partition("/")
                if beats.isdigit() and beat_type.isdigit():
                    return int(beats), int(beat_type)
            # `timeSignature/4` states only the denominator; 4/4 is the corpus default.
            if body.startswith("/") and body[1:].isdigit():
                return 4, int(body[1:])
    return 4, 4


@dataclass
class Inventory:
    crops: int = 0
    skipped: int = 0
    considered: int = 0
    agreeing: int = 0
    buckets: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict]] = field(default_factory=dict)

    def record(self, bucket: str, example: dict, keep: int = 60) -> None:
        self.buckets[bucket] += 1
        (
            self.examples.setdefault(bucket, []).append(example)
            if len(self.examples.get(bucket, [])) < keep
            else None
        )


def scan_crop(tokens_path: Path, inv: Inventory) -> None:
    sidecar_path = tokens_path.with_suffix(tokens_path.suffix + ".notation.json")
    if not sidecar_path.exists():
        return
    lines = [
        line.strip()
        for line in tokens_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))["notation"]

    notes: list[BeamableNote] = []
    engraved: list[tuple] = []
    onset = 0
    beats, beat_type = parse_metre(lines)
    entry = 0
    for line in lines:
        head = line.split()[0]
        if head == "barline":
            continue
        parsed = token_duration(head)
        if parsed is None:
            continue
        duration, flags = parsed
        members = line.count("&") + 1
        is_rest = head.startswith("rest_")
        ticks = int(duration * DIVISIONS)
        for member in range(members):
            if entry >= len(sidecar):
                inv.skipped += 1
                return
            # Every member of a chord shares the onset and the stem, so only the first
            # carries the beam decision - the rest would triple-count one choice.
            if member == 0:
                notes.append(
                    BeamableNote(onset=onset, duration=ticks, flags=flags, is_rest=is_rest)
                )
                engraved.append(tuple(BeamLevelState(state) for state in sidecar[entry]["beams"]))
            entry += 1
        onset += ticks
    if entry != len(sidecar):
        inv.skipped += 1
        return

    inv.crops += 1
    beat = beat_divisions(beats, beat_type, DIVISIONS)
    wide = wide_unit(beats, beat_type, DIVISIONS)
    predicted = automatic_beams(notes, beat, wide)
    for index, (note, engraved_vector, rule_vector) in enumerate(
        zip(notes, engraved, predicted, strict=True)
    ):
        if note.flags == 0 or note.is_rest:
            continue
        inv.considered += 1
        if tuple(rule_vector) == tuple(engraved_vector):
            inv.agreeing += 1
            continue
        example = {
            "crop": tokens_path.stem,
            "index": index,
            "rule": [str(s) for s in rule_vector],
            "engraved": [str(s) for s in engraved_vector],
        }
        rule_b, eng_b = is_beamed(rule_vector), is_beamed(engraved_vector)
        if eng_b and not rule_b:
            spans_rest = any(
                notes[j].is_rest for j in range(max(0, index - 3), min(len(notes), index + 4))
            )
            inv.record("beams_over_rest" if spans_rest else "engraved_beams", example)
        elif rule_b and not eng_b:
            inv.record("unbeamed_exception", example)
        elif any(
            (
                r == BeamLevelState.FLAG
                and e in {BeamLevelState.FORWARD_HOOK, BeamLevelState.BACKWARD_HOOK}
            )
            or (
                e == BeamLevelState.FLAG
                and r in {BeamLevelState.FORWARD_HOOK, BeamLevelState.BACKWARD_HOOK}
            )
            for r, e in zip(rule_vector, engraved_vector, strict=True)
            if r != e
        ):
            inv.record("hook_vs_flag", example)
        else:
            inv.record("different_grouping", example)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True, help="train/ or valid/ directory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(p for p in args.corpus.glob("*.txt") if p.name != "index.txt")
    if args.limit:
        paths = paths[: args.limit]
    inv = Inventory()
    for index, path in enumerate(paths, start=1):
        scan_crop(path, inv)
        if index % 2000 == 0 or index == len(paths):
            print(f"  [{index}/{len(paths)}] {inv.considered:,} notes", flush=True)

    disagreeing = inv.considered - inv.agreeing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "crops": inv.crops,
                "skipped_crops": inv.skipped,
                "considered": inv.considered,
                "agreeing": inv.agreeing,
                "disagreeing": disagreeing,
                "buckets": dict(inv.buckets.most_common()),
                "examples": inv.examples,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n{inv.crops:,} crops read, {inv.skipped:,} skipped on a count mismatch")
    print(
        f"{inv.considered:,} flagged notes, {inv.agreeing:,} agree "
        f"({inv.agreeing / max(inv.considered, 1):.1%}), {disagreeing:,} differ\n"
    )
    for name, count in inv.buckets.most_common():
        print(f"{name:<22} {count:>8,}  {count / max(disagreeing, 1):>7.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
