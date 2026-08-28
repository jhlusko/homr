"""Which symbols does a benchmark demand that the training corpus barely supplies?

Generalises the check that found the tuplet gap. The corpus carries 1.78% tuplet notes
against OSSQ's 6.58%, a 3.7x shortfall in exactly the class where the model makes 84% of
its rhythm errors and where fine-tuning has never helped. That gap was invisible to every
accuracy metric and obvious the moment supply was set beside demand.

So do it for every token, on every branch, and rank by what the shortfall COSTS: a symbol
the benchmark uses often and the corpus rarely contains is worth more than a rarer one
with a bigger ratio. `demand x log(demand/supply)` orders by that rather than by ratio
alone, which would put a single freak token at the top.

Four corpus fixes have now each removed a real defect of under 1% of tokens and moved
nothing measurable, so the useful question has stopped being "what is wrong with the
labels" and become "what is missing from them".

The first run of this found naturals: the corpus contains ZERO `N` lifts because
`build_clean_stage2_pairs` calls `strip_naturals`, which converts every one to empty
unconditionally, while OSSQ's references carry 879 (3.25% of lift tokens). No checkpoint
has ever predicted a natural - the base model included - so the lift branch has a hard
ceiling of 96.75% and naturals are roughly 40% of its remaining error.
"""

# flake8: noqa: T201

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

PAD = "\x00"
BRANCHES = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


def real(seq):
    return [t for t in seq if not t.startswith(PAD)]


def from_manifest(path: Path) -> dict[str, Counter]:
    """Supply, counted the way the model sees it.

    Read the .tokens file directly and the comparison is invalid twice over: simultaneous
    symbols share a line joined by "&", so whitespace splitting counts only the first
    member; and the `chord` separator does not exist in the file at all - it is
    materialised by `read_tokens`, which is also what produced the reference side of every
    scored .jsonl. Parsing the raw text made `chord` and `position=lower` look absent from
    a corpus full of both.
    """
    from training.transformer.training_vocabulary import read_tokens

    counts = {b: Counter() for b in BRANCHES}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        for symbol in read_tokens(line.split(",", 1)[1]):
            for branch in BRANCHES:
                counts[branch][str(getattr(symbol, branch))] += 1
    return counts


def from_jsonl(path: Path) -> tuple[dict[str, Counter], dict[str, Counter], dict[str, Counter]]:
    """Demand, plus how often the model actually gets each token right.

    A supply gap alone is not evidence of a problem. Staccato is 11.6x undersupplied on
    OSSQ and the model reads it at 84% recall - it learned the symbol elsewhere and the
    gap costs nothing. Naturals are absent from the corpus AND predicted 0 times. Only the
    intersection is actionable, so recall is reported beside the ratio.
    """
    demand = {b: Counter() for b in BRANCHES}
    hits = {b: Counter() for b in BRANCHES}
    predicted = {b: Counter() for b in BRANCHES}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for branch in BRANCHES:
            want = real(row.get(f"{branch}_reference", []))
            got = real(row.get(f"{branch}_predicted", []))
            for token in want:
                demand[branch][token] += 1
            for token in got:
                predicted[branch][token] += 1
            for a, b in zip(want, got):
                if a == b:
                    hits[branch][a] += 1
    return demand, hits, predicted


def octave(token: str) -> str:
    """Group pitches by octave; individual pitches are too sparse to compare."""
    m = re.match(r"^([A-G])[#bN]*([0-9])$", token)
    return f"octave {m.group(2)}" if m else token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, required=True, help="training manifest")
    parser.add_argument("--bench", action="append", required=True, metavar="LABEL=PATH",
                        help="a scored .jsonl; its REFERENCE side is the demand")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    supply = from_manifest(args.train)
    report = {}
    for spec in args.bench:
        label, _, path = spec.partition("=")
        demand, hits, predicted = from_jsonl(Path(path))
        print(f"\n=== {label} ===")
        rows = []
        for branch in BRANCHES:
            s, d = supply[branch], demand[branch]
            h, pr = hits[branch], predicted[branch]
            if branch == "pitch":
                s = Counter({octave(k): v for k, v in s.items()})
                d = Counter({octave(k): v for k, v in d.items()})
                h = Counter({octave(k): v for k, v in h.items()})
                pr = Counter({octave(k): v for k, v in pr.items()})
            s_total, d_total = sum(s.values()) or 1, sum(d.values()) or 1
            for token, n in d.items():
                if token in {"_", ".", "nonote"} or n < 20:
                    continue
                d_rate = n / d_total
                s_rate = s.get(token, 0) / s_total
                # +1e-6 so a token entirely absent from the corpus scores, rather than
                # dividing by zero and being skipped - absence is the strongest gap there is.
                ratio = d_rate / max(s_rate, 1e-6)
                if ratio <= 1.2:
                    continue
                recall = h.get(token, 0) / n
                # Rank by what the gap COSTS: demand, scaled by the shortfall, and by how
                # badly the model does on it. A token it already reads well is not a
                # problem however undersupplied it is.
                cost = d_rate * math.log(ratio) * (1 - recall)
                rows.append((cost, branch, token, d_rate, s_rate, ratio, n, recall,
                             pr.get(token, 0)))
        rows.sort(reverse=True)
        print(f"{'branch':>13} {'token':>20} {'demand':>8} {'supply':>8} {'ratio':>7} "
              f"{'count':>7} {'recall':>8} {'predicted':>10}")
        for _, branch, token, d_rate, s_rate, ratio, n, recall, npred in rows[:args.top]:
            supply_str = f"{100*s_rate:6.3f}%" if s_rate else "  ABSENT"
            print(f"{branch:>13} {token:>20} {100*d_rate:7.3f}% {supply_str} "
                  f"{ratio:6.1f}x {n:7,} {100*recall:7.1f}% {npred:10,}")
        report[label] = [
            {"branch": b, "token": t, "demand": d, "supply": s, "ratio": r,
             "count": n, "recall": rec, "predicted": npred}
            for _, b, t, d, s, r, n, rec, npred in rows[:args.top]
        ]
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
