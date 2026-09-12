"""How often does the beam head emit a sequence no engraver could draw?

`homr/transformer/beam_validation.py` answers this per voice and has never been run on
output: it sits in the shipping package with one test and no caller. The question it
settles is not accuracy. A per-note accuracy figure treats each vector independently,
which is exactly how the head predicts them, so a group that begins and never ends scores
as a handful of ordinary mistakes rather than as a construction that cannot be drawn.

Run over a dumped prediction file, so it costs no inference. Both sides are audited,
because the reference is the only available control: if the engraved vectors score badly
under the same processing, the head's figure is measuring the processing.

They do score badly - 6.2% of reference staves carry an interior finding - and that is
**not** explained here. The likeliest cause is that a crop can hold more than one voice
while `validate_voice` assumes one, so two voices' groups interleave and read as opened or
closed out of order; confirming it needs voice-separated vectors, which this dump does not
carry. So the absolute rates are an upper bound on both sides and the comparison is what
means anything.

The one finding the control does not muddy is `nested`, a BEGIN inside an open group: 12
in the reference against 682 from the head. Whatever inflates the reference's other
categories would inflate this one too, and does not.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from pathlib import Path

from homr.transformer.beam_validation import validate_voice
from homr.transformer.structured_notation import BeamLevelState

KINDS = (
    "unopened",
    "unclosed",
    "nested",
    "hook_at_primary_level",
    "single_note_groups",
)


def audit(path: Path, field: str, levels: int) -> tuple[Counter, int, int, int]:
    counts: Counter[str] = Counter()
    staves = invalid = groups = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            vectors = record.get(field)
            if not vectors:
                continue
            try:
                parsed = [tuple(BeamLevelState(state) for state in vector) for vector in vectors]
            except ValueError:
                continue
            findings = validate_voice(parsed, levels)
            staves += 1
            groups += findings.groups
            # A crop is one staff cut out of a system, so a group crossing either edge
            # is *correctly* open there - and it leaves a whole run, not one note: a
            # group that began before the crop shows as every CONTINUE up to its END.
            # The excluded region is therefore everything before the first BEGIN and
            # everything after the last END, which is exactly the span whose opening or
            # closing was cut away. Charging the head for that would charge it for the
            # corpus's framing rather than its own output.
            first_begin = next(
                (i for i, v in enumerate(parsed) if v and v[0] == BeamLevelState.BEGIN),
                len(parsed),
            )
            last_end = next(
                (
                    i
                    for i in range(len(parsed) - 1, -1, -1)
                    if parsed[i] and parsed[i][0] == BeamLevelState.END
                ),
                -1,
            )
            interior = {
                kind: [
                    index for index in getattr(findings, kind) if first_begin <= index <= last_end
                ]
                for kind in KINDS
            }
            if any(interior.values()):
                invalid += 1
            for kind in KINDS:
                counts[kind] += len(interior[kind])
                counts[kind + "_at_edge"] += len(getattr(findings, kind)) - len(interior[kind])
    return counts, staves, invalid, groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    for field in ("reference", "predicted"):
        counts, staves, invalid, groups = audit(args.predictions, field, args.levels)
        label = "engraved reference" if field == "reference" else "beam head"
        print(f"\n{label}: {staves:,} staves, {groups:,} beam groups")
        if staves:
            print(
                f"  staves containing something unengravable: {invalid:,} ({invalid / staves:.1%})"
            )
        for kind in KINDS:
            if counts[kind]:
                print(f"    {kind:<24} {counts[kind]:,}")
        if not any(counts[kind] for kind in KINDS):
            print("    none")
        edge = sum(counts[kind + "_at_edge"] for kind in KINDS)
        if edge:
            print(f"    ({edge:,} more at a crop edge, excluded - see the comment above)")


if __name__ == "__main__":
    main()
