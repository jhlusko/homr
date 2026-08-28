"""Profile a pair corpus: what is actually in it, and what is wrong with it.

Written because the training signal cannot settle corpus questions. Two runs of the same
corpus differ by 4.06pp on the independent benchmark, which is larger than every corpus
effect measured - so a defect has to be demonstrated from the data itself, not inferred
from a score.

Reports the things that make a corpus smaller or more skewed than its pair count
suggests:

* **Duplicate labels.** Identical token streams train as one example with extra weight.
  A corpus of 4,000 pairs holding 3,000 distinct labels is a 3,000-pair corpus with a
  bias toward whatever repeats.
* **Trivial pairs.** A label of two or three symbols carries almost no supervision but
  occupies a slot in every batch.
* **Silent pairs.** All-rest labels teach the model that a staff of that shape is empty,
  which is right for a vocal line under a piano introduction and wrong nearly everywhere
  else.
* **Score concentration.** If a tenth of the scores supply half the pairs, held-out
  score-disjoint validation is measuring generalisation across far fewer effective
  units than the pair count implies.
* **Shape against the benchmark.** If the corpus is mostly grand staves and the
  benchmark is mostly single staves, training on it cannot be expected to move the
  benchmark - a structural mismatch no amount of label cleaning fixes.
"""

# flake8: noqa: T201

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from training.omr_datasets.audit_label_consistency import is_single_staff
from training.transformer.training_vocabulary import read_tokens

MEASURE_DIVIDERS = frozenset(
    {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
)
#: Below this a label carries too little to supervise anything - a clef, a key and a
#: note or two.
TRIVIAL_SYMBOLS = 6


def score_of(stem: str) -> str:
    match = re.match(r"^(.+?)-sys\d+-v\d+$", stem)
    return match.group(1) if match else stem


def profile(manifest: Path, label: str) -> dict:
    rows = [line.split(",", 1) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    per_score: Counter = Counter()
    digests: Counter = Counter()
    bars: Counter = Counter()
    symbols: list[int] = []
    staff_kind: Counter = Counter()
    trivial = silent = unreadable = 0

    for image, tokens in rows:
        stem = Path(image).stem
        per_score[score_of(stem)] += 1
        try:
            syms = read_tokens(tokens)
        except Exception:  # noqa: BLE001
            unreadable += 1
            continue
        body = [s for s in syms if s.rhythm not in MEASURE_DIVIDERS]
        symbols.append(len(body))
        bars[sum(1 for s in syms if s.rhythm in MEASURE_DIVIDERS)] += 1
        staff_kind["single" if is_single_staff(syms) else "grand"] += 1
        digests[hashlib.sha256(
            "\n".join(f"{s.rhythm}|{s.pitch}|{s.position}" for s in syms).encode()
        ).hexdigest()] += 1
        if len(body) < TRIVIAL_SYMBOLS:
            trivial += 1
        if body and all(s.rhythm.startswith("rest") for s in body):
            silent += 1

    total = len(rows)
    distinct = len(digests)
    repeated = sum(count for count in digests.values() if count > 1)
    top_scores = per_score.most_common()
    half = 0
    covered = 0
    for _, count in top_scores:
        covered += count
        half += 1
        if covered >= total / 2:
            break
    symbols.sort()

    def pct(n: int) -> str:
        return f"{100 * n / max(total, 1):5.1f}%"

    print(f"\n=== {label} ===")
    print(f"{total:,} pairs across {len(per_score)} scores")
    print(f"  distinct labels           {distinct:,}  ({pct(distinct)} of pairs)")
    print(f"  pairs sharing a label     {repeated:,}  ({pct(repeated)})")
    print(f"  trivial (<{TRIVIAL_SYMBOLS} symbols)      {trivial:,}  ({pct(trivial)})")
    print(f"  all-rest labels           {silent:,}  ({pct(silent)})")
    print(f"  unreadable                {unreadable:,}")
    print(f"  staff type                single {staff_kind['single']:,} / grand {staff_kind['grand']:,}")
    if symbols:
        print(f"  symbols per pair          min {symbols[0]}  p25 {symbols[len(symbols)//4]}  "
              f"median {symbols[len(symbols)//2]}  p75 {symbols[3*len(symbols)//4]}  max {symbols[-1]}")
    print(f"  bars per pair             {dict(sorted(bars.items())[:8])}")
    print(f"  half the pairs come from  {half} score(s) ({100*half/max(len(per_score),1):.0f}% of scores)")
    print(f"  largest score             {top_scores[0][0]} with {top_scores[0][1]} pairs" if top_scores else "")
    return {
        "label": label, "pairs": total, "scores": len(per_score),
        "distinct_labels": distinct, "pairs_sharing_a_label": repeated,
        "trivial": trivial, "silent": silent, "unreadable": unreadable,
        "single_staff": staff_kind["single"], "grand_staff": staff_kind["grand"],
        "scores_for_half_the_pairs": half,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    out = []
    for spec in args.manifest:
        label, _, path = spec.partition("=")
        out.append(profile(Path(path), label))
    if args.report:
        args.report.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
