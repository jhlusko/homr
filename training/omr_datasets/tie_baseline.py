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

from training.omr_datasets.reference_label_audit import (
    _notes,
    _onsets,
    _same_pitch,
    _tie_partner_index,
    _voices,
)


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


def scan(tokens_path: Path, baseline: TieBaseline) -> bool:
    """Score one crop's tie labels against the constraint that defines a tie.

    The partner search is `reference_label_audit`'s, imported rather than reimplemented.
    The version that lived here carried `(pitch, is_rest, chord)` and nothing about staff
    or voice, so on a grand staff it hunted for a tie's partner in the other hand and on a
    polyphonic staff in the other voice. It also compared bare letter and octave, which
    lets E-flat4 pair with E-natural4. Those are the faults that made this baseline's
    Lieder reading (63.6%) a property of the tool rather than of the corpus.
    """
    sidecar_path = tokens_path.with_suffix(tokens_path.suffix + ".notation.json")
    if not sidecar_path.exists():
        return False
    notes = _notes(tokens_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))["notation"]
    if len(notes) != len(sidecar):
        return False
    voices, onsets = _voices(sidecar), _onsets(sidecar)

    for index, entry in enumerate(sidecar):
        if notes[index][1]:
            continue
        baseline.notes += 1
        state = entry.get("tie", "none")
        baseline.states[state] += 1

        partner = _tie_partner_index(notes, index, voices, onsets)
        joins_same_pitch = partner is not None and _same_pitch(notes, index, partner)

        if state in {"start", "start_and_stop"}:
            # A tie on the last note of its own line continues into the next system,
            # which is ordinary engraving: the partner is simply not in this image.
            # Counting it as a broken constraint would charge the corpus for its own
            # framing, the same way an unclosed beam at a crop edge is not unclosed.
            if partner is None:
                baseline.tie_across_crop_edge += 1
            else:
                baseline.tie_starts += 1
                if joins_same_pitch:
                    baseline.tie_same_pitch += 1
                elif len(baseline.violations) < 40:
                    baseline.violations.append(
                        {
                            "crop": tokens_path.stem,
                            "index": index,
                            "pitch": notes[index][0],
                            "next_pitch": notes[partner][0],
                            "state": state,
                        }
                    )

        if joins_same_pitch:
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

    # Both token suffixes: OSSQ writes `.txt`, the Lieder builder writes `.tokens`.
    # Globbing only `*.txt` is why this baseline's figures have always been OSSQ-only -
    # it read zero crops from Lieder and said so as "0 crops read", which is easy to miss.
    paths = sorted(
        p
        for suffix in ("*.txt", "*.tokens")
        for p in args.corpus.glob(suffix)
        if p.name != "index.txt"
    )
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
