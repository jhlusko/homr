"""What a rule already knows about ties, and whether the labels obey it.

The tie head is reported at macro-F1 .844 against nothing at all. `beam_baseline` exists
because a head that only reproduces what duration and metre already determine has learned
nothing; the same question has never been asked of ties, and for ties the derivable part is
unusually sharp.

A tie is not a curve that happens to look like a slur. `TieState`'s own docstring puts it
exactly: "a tie joins two notations of *one* pitch into a single sounding note, while a
slur groups distinct pitches under one phrase." So **same pitch is a necessary condition**,
which makes the rule a constraint rather than a prediction - a tie between different
pitches is not a close call, it is impossible, the way an unclosed beam is.

Two things are measured, and they answer different questions:

  `constraint`   Do the engraved labels obey it? If a reference tie joins two different
                 pitches, either the corpus is wrong or the reading of the label is, and
                 that has to be settled before the constraint is used to judge a head.

  `derivable`    Of adjacent same-pitch note pairs, how many are actually tied? This is
                 the ceiling for "tie whenever the pitch repeats" - the cheapest possible
                 predictor - and so the number a trained head has to beat to be worth its
                 parameters.

Run over the corrected scanned corpus, where each crop's tokens and engraved notation were
joined when the corpus was built.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TieBaseline:
    notes: int = 0
    #: Reference ties whose partner carries the same pitch, and those that do not.
    tie_starts: int = 0
    tie_same_pitch: int = 0
    #: Ties whose partner is in the next system, so this crop cannot show it.
    tie_across_crop_edge: int = 0
    violations: list[dict] = field(default_factory=list)
    #: Adjacent same-pitch pairs, and how many of them the engraving actually ties.
    repeated_pitch_pairs: int = 0
    repeated_and_tied: int = 0
    states: Counter[str] = field(default_factory=Counter)


def _notes_of(tokens_path: Path) -> list[tuple[str, bool, int]] | None:
    """(pitch, is_rest, chord id) per note, chord members expanded for the sidecar.

    The chord id matters for ties. A chord's members are consecutive in this list, so
    "the next note" for a chord member is its own sibling - the wrong candidate entirely,
    since a tie joins one notehead to a notehead in the *next* chord. Without the id, a
    chord's lower voice looks tied to its own upper voice and the constraint reports a
    violation the engraving never committed.
    """
    out: list[tuple[str, bool, int]] = []
    chord = 0
    for raw_line in tokens_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        members = line.split("&")
        if not members[0].split() or not members[0].split()[0].startswith(("note_", "rest_")):
            continue
        for member in members:
            fields = member.split()
            if not fields:
                continue
            head = fields[0]
            pitch = fields[1] if len(fields) > 1 else ""
            out.append((pitch, head.startswith("rest_"), chord))
        chord += 1
    return out


def scan(tokens_path: Path, baseline: TieBaseline) -> bool:
    sidecar_path = tokens_path.with_suffix(tokens_path.suffix + ".notation.json")
    if not sidecar_path.exists():
        return False
    notes = _notes_of(tokens_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))["notation"]
    if notes is None or len(notes) != len(sidecar):
        return False

    for index, ((pitch, is_rest, chord), entry) in enumerate(zip(notes, sidecar, strict=True)):
        if is_rest:
            continue
        baseline.notes += 1
        state = entry.get("tie", "none")
        baseline.states[state] += 1

        # The partner is a notehead in the *next* chord, not the rest of this one, and a
        # rest between them ends the sounding note so nothing can be tied across it.
        next_pitch = None
        for later in range(index + 1, len(notes)):
            if notes[later][2] == chord:
                continue
            if notes[later][1]:
                break
            # A chord can hold the same pitch in more than one voice; any member with
            # this pitch is a legitimate partner.
            candidates = [other[0] for other in notes[later:] if other[2] == notes[later][2]]
            next_pitch = pitch if pitch in candidates else candidates[0]
            break

        if state in {"start", "start_and_stop"}:
            # A tie on the last note of a crop continues into the next system, which is
            # ordinary engraving: the partner is simply not in this image. Counting it as
            # a broken constraint would charge the corpus for its own framing, the same
            # way an unclosed beam at a crop edge is not an unclosed beam.
            if next_pitch is None:
                baseline.tie_across_crop_edge += 1
            else:
                baseline.tie_starts += 1
                if next_pitch == pitch:
                    baseline.tie_same_pitch += 1
                elif len(baseline.violations) < 40:
                    baseline.violations.append(
                        {
                            "crop": tokens_path.stem,
                            "index": index,
                            "pitch": pitch,
                            "next_pitch": next_pitch,
                            "state": state,
                        }
                    )

        if next_pitch is not None and next_pitch == pitch:
            baseline.repeated_pitch_pairs += 1
            if state in {"start", "start_and_stop"}:
                baseline.repeated_and_tied += 1
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(p for p in args.corpus.glob("*.txt") if p.name != "index.txt")
    if args.limit:
        paths = paths[: args.limit]
    baseline = TieBaseline()
    read = skipped = 0
    for index, path in enumerate(paths, start=1):
        if scan(path, baseline):
            read += 1
        else:
            skipped += 1
        if index % 5000 == 0 or index == len(paths):
            print(f"  [{index}/{len(paths)}] {baseline.notes:,} notes", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(vars(baseline), indent=1, default=str), encoding="utf-8")

    starts = baseline.tie_starts
    pairs = baseline.repeated_pitch_pairs
    print(f"\n{read:,} crops read, {skipped:,} skipped; {baseline.notes:,} pitched notes")
    print(f"tie states: {dict(baseline.states)}")
    print("\nconstraint - a tie joins one pitch:")
    print(f"  ({baseline.tie_across_crop_edge:,} ties continue past the crop, excluded)")
    if starts:
        print(
            f"  {baseline.tie_same_pitch:,} of {starts:,} reference ties "
            f"({baseline.tie_same_pitch / starts:.1%}) join a repeat of the same pitch"
        )
        print(f"  {starts - baseline.tie_same_pitch:,} do not")
    print("\nderivable - tie whenever the pitch repeats:")
    if pairs:
        print(
            f"  {baseline.repeated_and_tied:,} of {pairs:,} adjacent same-pitch pairs "
            f"are tied ({baseline.repeated_and_tied / pairs:.1%})"
        )
        print(
            "  that share is the precision of the cheapest predictor, and the number a"
            "\n  trained head has to beat to earn its parameters"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
