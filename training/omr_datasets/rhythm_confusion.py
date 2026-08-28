"""Which note values does the model confuse, and with what?

The error taxonomy says rhythm is the largest remaining failure class - 39 of 148 dense
staves for the best checkpoint, against 10 length and 6 structural - and that fine-tuning
barely touches it (44 -> 39). Nothing has looked at *which* durations go wrong.

The answer decides where effort goes, because the candidate causes need different fixes:

* **dots** - `4.` read as `4`, or the reverse. A dot is a few pixels beside a notehead;
  this would be a resolution or augmentation problem.
* **beam counts** - `8` read as `16`. One beam versus two, which the structured beam
  heads exist to help with.
* **tuplets** - `12` (a triplet eighth) read as `8`. The page often marks these only with
  a bracket or numeral, and where it marks them not at all the label cannot be right
  either; 417 pairs were quarantined over exactly this.
* **gross** - a whole confused with an eighth, which would mean something else is wrong.

Aligns the two streams position by position and only compares where they have not already
diverged in length, so a single insertion does not manufacture a hundred false confusions.
"""

# flake8: noqa: T201

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PAD = "\x00"


def real(seq):
    return [t for t in seq if not t.startswith(PAD)]


def note_value(token: str) -> str | None:
    """`note_4.` -> `4.`; anything that is not a note returns None."""
    m = re.match(r"^note_([0-9]+\.?)$", token)
    return m.group(1) if m else None


def kind(want: str, got: str) -> str:
    w, g = want.rstrip("."), got.rstrip(".")
    if w == g:
        return "dot only"
    try:
        wi, gi = int(w), int(g)
    except ValueError:
        return "other"
    # Triplet values in this vocabulary are 3x the plain value (12 = triplet eighth).
    if wi == 3 * gi or gi == 3 * wi or {wi, gi} in ({8, 12}, {16, 24}, {4, 6}):
        return "tuplet vs plain"
    if wi == 2 * gi or gi == 2 * wi:
        return "one beam / halving"
    return "gross"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--min-symbols", type=int, default=45)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        confusion: Counter = Counter()
        kinds: Counter = Counter()
        compared = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            want = real(row.get("rhythm_reference", []))
            got = real(row.get("rhythm_predicted", []))
            if len(want) < args.min_symbols:
                continue
            # Only compare while the streams are the same length; past a length
            # divergence, position-wise comparison is meaningless.
            for a, b in zip(want, got):
                va, vb = note_value(a), note_value(b)
                if va is None or vb is None:
                    continue
                compared += 1
                if va != vb:
                    confusion[(va, vb)] += 1
                    kinds[kind(va, vb)] += 1
        total = sum(confusion.values())
        print(f"\n=== {label} ===")
        print(f"{compared:,} note values compared, {total:,} wrong "
              f"({100 * total / max(compared, 1):.2f}%)")
        for k, v in kinds.most_common():
            print(f"   {k:>18}: {v:5d}  ({100 * v / max(total, 1):4.1f}%)")
        print("   most confused pairs (reference -> predicted):")
        for (a, b), v in confusion.most_common(8):
            print(f"      {a:>5} -> {b:<5}  {v}")
        report[label] = {"compared": compared, "wrong": total,
                         "kinds": dict(kinds),
                         "pairs": {f"{a}->{b}": v for (a, b), v in confusion.most_common(20)}}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
