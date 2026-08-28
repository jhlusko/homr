"""Break a checkpoint's errors into kinds, instead of one accuracy number.

Token accuracy treats every mistake alike: a dropped barline, a wrong pitch and a
wholesale divergence all subtract from the same total. That has been enough to rank
checkpoints and useless for saying *what* got better - and with a 4pp seed spread on the
aggregate, "what changed" is often more answerable than "how much".

Four kinds, chosen because each implies a different fix:

* **structural** - the prediction has a different number of measure dividers from the
  reference. The model lost the bar grid, which is an alignment-shaped failure and the
  one that makes a transcription unusable rather than merely wrong.
* **length** - same bar count, different symbol count. Notes invented or dropped inside
  a correct grid.
* **pitch-only** - same length, same rhythm, wrong pitches. A reading error on the
  staff, not a rhythmic one.
* **rhythm** - same length, wrong durations.

A staff can qualify for several; the first that applies wins, most-severe first, so the
counts partition the corpus.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from pathlib import Path

PAD = "\x00"
DIVIDERS = ("barline", "doublebarline", "bolddoublebarline",
            "repeatStart", "repeatEnd", "repeatBoth")


def real(seq):
    return [t for t in seq if not t.startswith(PAD)]


def classify(row) -> tuple[str, int]:
    ref_r, got_r = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
    ref_p, got_p = real(row.get("pitch_reference", [])), real(row.get("pitch_predicted", []))
    bars_ref = sum(1 for t in ref_r if t in DIVIDERS)
    bars_got = sum(1 for t in got_r if t in DIVIDERS)
    if bars_ref != bars_got:
        return "structural", abs(bars_ref - bars_got)
    if len(ref_r) != len(got_r):
        return "length", abs(len(ref_r) - len(got_r))
    rhythm_wrong = sum(1 for a, b in zip(ref_r, got_r) if a != b)
    pitch_wrong = sum(1 for a, b in zip(ref_p, got_p) if a != b)
    if rhythm_wrong:
        return "rhythm", rhythm_wrong
    if pitch_wrong:
        return "pitch-only", pitch_wrong
    return "exact", 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--min-symbols", type=int, default=0,
                        help="restrict to staves at least this dense; the dense cut is "
                             "where the measurement is stable (see BENCHMARKS.md)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        rows = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["tokens"]] = row
        runs[label] = rows

    shared = sorted(set.intersection(*(set(r) for r in runs.values())))
    if args.min_symbols:
        first = next(iter(runs.values()))
        shared = [i for i in shared
                  if len(real(first[i].get("rhythm_reference", []))) >= args.min_symbols]
    print(f"{len(shared):,} staves scored by all runs"
          + (f", at least {args.min_symbols} symbols" if args.min_symbols else ""))

    kinds = ["exact", "pitch-only", "rhythm", "length", "structural"]
    header = f"{'run':<18}" + "".join(f"{k:>13}" for k in kinds)
    print("\n" + header)
    report = {}
    for label, rows in runs.items():
        counts: Counter = Counter()
        for i in shared:
            kind, _ = classify(rows[i])
            counts[kind] += 1
        report[label] = dict(counts)
        print(f"{label:<18}" + "".join(
            f"{counts[k]:>7} {100 * counts[k] / max(len(shared), 1):>4.1f}%" for k in kinds))

    base = next(iter(runs))
    print(f"\nchange against {base}, in staves")
    for label in list(runs)[1:]:
        deltas = " ".join(
            f"{k}: {report[label].get(k, 0) - report[base].get(k, 0):+d}" for k in kinds)
        print(f"  {label:<16} {deltas}")

    if args.report:
        args.report.write_text(json.dumps(
            {"staves": len(shared), "min_symbols": args.min_symbols, "runs": report},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
