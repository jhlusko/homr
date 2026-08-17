"""
Duplicate the worst-performing scores' staves in a training index, rather than filtering
or transforming any image.

27.60 found the scan gap concentrated by score - a 70x spread in collapse rate between the
best and the worst of nine documents - and 27.63 ruled out a page-level fix: CLAHE contrast
normalisation damages the crisp scans faster than it helps the faint ones, because a faint
page has little local structure for it to exploit. 27.60's own shape points at a different
lever: change *which documents* the model sees more of, not what each image looks like.

**Oversampling, not filtering.** Dropping the worst scores would shrink an already-small
scanned corpus and remove the exact examples a deployed system will meet - a scan that reads
like sq8806881 is not hypothetical, it is what a phone photograph of a page looks like.
Repeating their lines in the index gives the loss more chances to fit them without touching
the paired synthetic set or discarding anything.

**Weighted by collapse rate, not by a fixed multiplier.** A score with a 21.9% collapse rate
needs more repetition than one at 5.4%, and a fixed factor applied to every "bad" score
would either under-correct the worst or over-correct the mild ones. The weight is linear in
collapse rate above a floor, so a score with no collapsed staves keeps its natural weight of
one line.

This changes an index file, which is data, not a model - no GPU needed to build or verify it,
only to find out whether it helps.
"""

# flake8: noqa: T201

import argparse
import collections
from dataclasses import dataclass
from pathlib import Path

from training.transformer.domain_gap import Pair, pair_up, staff_accuracy


@dataclass(frozen=True)
class ScoreWeight:
    score: str
    collapse_rate: float
    repeats: int


def collapse_rates(pairs: list[Pair], threshold: float = 0.5) -> dict[str, float]:
    """Share of each score's staves whose beam accuracy drops by more than `threshold`."""
    by_score: dict[str, list[Pair]] = collections.defaultdict(list)
    for pair in pairs:
        by_score[pair.name.split("_")[0]].append(pair)
    return {
        score: sum(1 for p in staves if p.drop > threshold) / len(staves)
        for score, staves in by_score.items()
    }


def repeat_counts(
    rates: dict[str, float], floor: float = 0.1, max_repeats: int = 6
) -> dict[str, ScoreWeight]:
    """How many times each score's lines should appear in the reweighted index.

    Linear above a floor rather than proportional from zero: a score just over the floor
    should not be treated as though it were clean, but the floor keeps sampling noise in a
    low but nonzero collapse rate from producing a meaningless x1.02 repeat. Capped, for the
    same reason the loss weight in 27.50 was capped - an extreme multiplier turns a handful
    of documents into most of an epoch, which risks memorising those documents rather than
    learning what makes them hard.

    **Scaled against the worst rate actually observed, not against a hypothetical 100%
    collapse.** The first version scaled toward rate=1.0, and on nine real scores whose
    worst collapse rate is 21.9% that put `max_repeats` at a point nothing reaches - eight
    of nine scores landed inside rounding distance of x1 and only the single worst score
    moved at all. A scale calibrated to data no score produces is a scale that does not
    fire on the data that exists.
    """
    if not rates:
        return {}
    worst = max(rates.values())
    weights = {}
    for score, rate in rates.items():
        if rate <= floor or worst <= floor:
            repeats = 1
        else:
            span = max(1e-9, worst - floor)
            repeats = 1 + round((max_repeats - 1) * (rate - floor) / span)
        weights[score] = ScoreWeight(score, rate, min(max_repeats, max(1, repeats)))
    return weights


def reweight_index(
    index: Path, weights: dict[str, ScoreWeight], out: Path
) -> tuple[int, int]:
    """Write a new index with each line repeated by its score's weight.

    Lines for scores absent from `weights` - not part of the measured comparison, or with
    no staves in it - are kept once. Silently repeating an unmeasured score would be
    guessing at a weight rather than acting on one.
    """
    lines = index.read_text(encoding="utf-8").splitlines()
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for line in lines:
            if not line.strip():
                continue
            tokens_path = line.split(",")[-1]
            score = Path(tokens_path).name.split("_")[0]
            repeats = weights[score].repeats if score in weights else 1
            for _ in range(repeats):
                handle.write(line + "\n")
                written += 1
    return len(lines), written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--synthetic-predictions", type=Path, required=True)
    parser.add_argument("--scanned-predictions", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="The training index to reweight.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor", type=float, default=0.1)
    parser.add_argument("--max-repeats", type=int, default=6)
    args = parser.parse_args()

    pairs = pair_up(
        staff_accuracy(args.synthetic_predictions), staff_accuracy(args.scanned_predictions)
    )
    rates = collapse_rates(pairs)
    weights = repeat_counts(rates, args.floor, args.max_repeats)

    print(f"collapse rate and repeat count, {len(weights)} score(s):")
    for score, weight in sorted(weights.items(), key=lambda kv: -kv[1].collapse_rate):
        print(f"  {score:<12} {weight.collapse_rate:>6.1%}  x{weight.repeats}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    before, after = reweight_index(args.index, weights, args.out)
    print(f"\n{before:,} lines -> {after:,} lines ({args.out})")


if __name__ == "__main__":
    main()
