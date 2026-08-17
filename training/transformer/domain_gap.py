"""
Is the synthetic-to-scan gap spread over every staff, or concentrated in some of them?

27.38 measured the gap and 27.58 showed what it costs: on scans both structured heads are net
regressions against their rules. The reframing that follows - close the gap rather than add
heads - is only actionable once the gap's shape is known, and there are two shapes it could
have:

  * **spread** - every staff degrades a little, because scanned images are genuinely harder
    to read. Then the fix is visual: augmentation, more scanned data, a better encoder.
  * **concentrated** - most staves are as good as their synthetic twin and a minority
    collapse. Then the likely cause is not difficulty but **misalignment**, because 27.14
    measured scanned staff detection reporting five to nine staves in a four-part system, and
    a miscount shifts every crop-to-part pairing after it. That is a label problem wearing a
    domain problem's clothes, and augmentation would not touch it.

The two are distinguishable because the scanned and synthetic tracks share their token files -
`sq8907120_0001_0001_1.txt` names the same staff in both - so each staff can be scored twice
and the pairs compared. A staff at 95% synthetic and 20% scanned is a broken crop. The same
staff at 95% and 75% is the domain.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import statistics
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pair:
    """One staff's beam accuracy under both renderings."""

    name: str
    synthetic: float
    scanned: float
    notes: int

    @property
    def drop(self) -> float:
        return self.synthetic - self.scanned


def staff_accuracy(
    predictions: Path, field: str = "reference"
) -> dict[str, tuple[float, int]]:
    """Per token file, the share of positions whose whole vector for `field` is right.

    Keyed by the token file's name rather than its path, because the two tracks write the
    same names into different directories - which is exactly what makes them comparable.

    `field` selects which head's vectors to compare: "reference" reads the beam pair
    (`reference`/`predicted`), and any other value reads `{field}_reference` /
    `{field}_predicted` - so `field="slur"` reads the slur vectors 27.60 originally had no
    way to ask this question of. Every vector is a list, and Python's list equality treats
    it as one unit, which is what "does this note's whole state match" means for either head.
    """
    ref_key = "reference" if field == "reference" else f"{field}_reference"
    pred_key = "predicted" if field == "reference" else f"{field}_predicted"

    found: dict[str, tuple[float, int]] = {}
    with predictions.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reference = record.get(ref_key)
            predicted = record.get(pred_key)
            if not reference or not predicted:
                continue
            matched = sum(1 for want, got in zip(reference, predicted) if want == got)
            total = min(len(reference), len(predicted))
            if total:
                found[Path(record["tokens"]).name] = (matched / total, total)
    return found


def pair_up(synthetic: dict, scanned: dict) -> list[Pair]:
    shared = sorted(set(synthetic) & set(scanned))
    return [
        Pair(name, synthetic[name][0], scanned[name][0], synthetic[name][1])
        for name in shared
    ]


def describe(pairs: list[Pair]) -> str:
    if not pairs:
        return "no staves in common - do the two prediction files name the same token files?"

    drops = sorted(pair.drop for pair in pairs)
    collapsed = [pair for pair in pairs if pair.drop > 0.5]
    steady = [pair for pair in pairs if pair.drop <= 0.1]

    # How much of the total loss the worst tenth accounts for. Spread and concentrated
    # differ here more sharply than in any average: a uniform gap gives about 10%.
    worst = sorted(pairs, key=lambda pair: -pair.drop)
    tenth = max(1, len(pairs) // 10)
    total_loss = sum(pair.drop * pair.notes for pair in pairs)
    worst_loss = sum(pair.drop * pair.notes for pair in worst[:tenth])

    lines = [
        f"{len(pairs):,} staves scored under both renderings",
        f"  mean accuracy  synthetic {statistics.mean(p.synthetic for p in pairs):.1%}"
        f"   scanned {statistics.mean(p.scanned for p in pairs):.1%}",
        "",
        "drop per staff (synthetic minus scanned):",
        f"  median {statistics.median(drops):.1%}"
        f"   quartiles {drops[len(drops) // 4]:.1%} / {drops[3 * len(drops) // 4]:.1%}",
        f"  unchanged or nearly so (<= 10 points): {len(steady):,} ({len(steady) / len(pairs):.1%})",
        f"  collapsed (> 50 points):               {len(collapsed):,}"
        f" ({len(collapsed) / len(pairs):.1%})",
        "",
        f"share of all lost notes falling in the worst 10% of staves: {worst_loss / max(1e-9, total_loss):.1%}",
        "  a gap spread evenly over every staff would put about 10% here;",
        "  a gap caused by broken crops would put most of it here.",
    ]
    if collapsed:
        lines.append("")
        lines.append("worst staves:")
        for pair in worst[:6]:
            lines.append(
                f"  {pair.name}: {pair.synthetic:.0%} -> {pair.scanned:.0%} ({pair.notes} notes)"
            )
        # Reported as a rate per score, not a count of scores. The first version of this
        # line said "the 326 collapsed staves come from 9 scores", which reads as clustering
        # and is not: the validation set has exactly 9 scores, so that is all of them. A
        # count is only evidence of clustering against the total it is drawn from.
        every = collections.Counter(pair.name.split("_")[0] for pair in pairs)
        failing = collections.Counter(pair.name.split("_")[0] for pair in collapsed)
        lines.append("")
        lines.append(f"collapse rate by score ({len(every)} score(s) in this split):")
        for name, count in sorted(every.items(), key=lambda kv: -failing[kv[0]] / kv[1]):
            lines.append(f"  {name}: {failing[name]:>4}/{count:<4} ({failing[name] / count:.1%})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--synthetic", type=Path, required=True, help="predictions.jsonl")
    parser.add_argument("--scanned", type=Path, required=True)
    parser.add_argument(
        "--field", default="reference",
        help="'reference' for beams (default), or a head name like 'slur' to read "
        "{field}_reference/{field}_predicted.",
    )
    args = parser.parse_args()

    print(describe(pair_up(
        staff_accuracy(args.synthetic, args.field), staff_accuracy(args.scanned, args.field)
    )))


if __name__ == "__main__":
    main()
