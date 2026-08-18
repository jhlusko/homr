"""
Does the frozen core's own note-reading degrade on scans - independent of every structured
head this design has trained?

27.78 corrected a claim made without checking: run 426, the exact checkpoint every phase
script in this design fine-tunes, was already trained on lieder+grandstaff+primus+pdmx+
musetrainer (Training.md) before this design existed. Every domain-gap figure recorded here
since 27.38 measures the *structured heads* - beam, stem, tie, the slotted slur - which had
no training data at all before this project built them. None of it measures whether the
frozen core's own predictions, the fields it was already pretrained on, hold up on scans.

This closes that gap using `validation/ossq.py` unmodified - homr's plain pipeline, segnet
plus the base decoder, zero structured heads involved - scored on the same OSSQ scanned and
synthetic validation splits used throughout. Its own SQLite output already gives per-page
NED, overall and per component (rhythm, pitch, lift, articulation, slur - the six original
`EncodedSymbol` fields this design never touched), keyed by `sample_id` = `{score}_{page}`.
Synthetic and scanned share those ids for the same page, which is what makes them
comparable - the same principle `training/transformer/domain_gap.py` used for the
structured heads' predictions, applied here to a completely independent measurement.

**Lower NED is better**, the opposite sign convention from `domain_gap.py`'s accuracy
figures. "Drop" here means NED getting worse (higher) on scans, not lower.
"""

# flake8: noqa: T201

import argparse
import collections
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

#: The overall figure plus every component NED the samples table stores.
COMPONENTS = ("ned", "rhythm_ned", "pitch_ned", "lift_ned", "articulation_ned", "slur_ned")


def read_scores(db_path: Path) -> dict[str, dict[str, float]]:
    """Every scored sample, keyed by id, each a dict of component -> NED (0-1 scale).

    Failed samples (`error` set, no `ned`) are skipped rather than treated as 0% or 100% -
    a page the tool could not process at all is not comparable to one it read and scored.

    **Values are already 0-1 fractions in the database, not percentages.** The first
    version of this function divided by 100 on the assumption they were stored as
    percentages like the tool's own printed log lines (`NED=  2.6%`) - the log formats a
    fraction for display, the database keeps the fraction itself. That bug produced a mean
    NED of 0.1% on data whose printed samples ranged from 2% to 14%, caught by checking the
    raw column values directly against the log rather than trusting the first result.
    """
    connection = sqlite3.connect(str(db_path))
    columns = ", ".join(COMPONENTS)
    rows = connection.execute(f"SELECT sample_id, {columns} FROM samples WHERE ned IS NOT NULL")
    found = {}
    for row in rows:
        sample_id, *values = row
        found[sample_id] = dict(zip(COMPONENTS, values))
    connection.close()
    return found


@dataclass(frozen=True)
class Pair:
    sample_id: str
    synthetic: float
    scanned: float

    @property
    def degradation(self) -> float:
        """How much worse (higher) NED gets on scans. Positive means scans are harder."""
        return self.scanned - self.synthetic


def pair_up(
    synthetic: dict[str, dict[str, float]], scanned: dict[str, dict[str, float]], component: str
) -> list[Pair]:
    shared = sorted(set(synthetic) & set(scanned))
    return [Pair(sid, synthetic[sid][component], scanned[sid][component]) for sid in shared]


def describe(pairs: list[Pair], component: str) -> str:
    if not pairs:
        return "no pages in common - do the two databases name the same sample ids?"

    degradations = sorted(pair.degradation for pair in pairs)
    worst = sorted(pairs, key=lambda pair: -pair.degradation)
    tenth = max(1, len(pairs) // 10)
    total_degradation = sum(max(0.0, pair.degradation) for pair in pairs)
    worst_degradation = sum(max(0.0, pair.degradation) for pair in worst[:tenth])

    lines = [
        f"{component}: {len(pairs):,} pages scored under both renderings",
        f"  mean NED  synthetic {statistics.mean(p.synthetic for p in pairs):.1%}"
        f"   scanned {statistics.mean(p.scanned for p in pairs):.1%}",
        "",
        "  NED increase per page, scanned minus synthetic (positive = scans harder):",
        f"    median {statistics.median(degradations):.1%}"
        f"   quartiles {degradations[len(degradations) // 4]:.1%} /"
        f" {degradations[3 * len(degradations) // 4]:.1%}",
        f"  share of all degradation in the worst 10% of pages:"
        f" {worst_degradation / max(1e-9, total_degradation):.1%}",
    ]

    by_score: dict[str, list[Pair]] = collections.defaultdict(list)
    for pair in pairs:
        by_score[pair.sample_id.split("_")[0]].append(pair)
    if len(by_score) > 1:
        lines.append("")
        lines.append(f"  by score ({len(by_score)} scores):")
        rates = {
            score: statistics.mean(p.degradation for p in ps) for score, ps in by_score.items()
        }
        for score, rate in sorted(rates.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {score:<14} mean NED increase {rate:>+7.1%}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--synthetic", type=Path, required=True, help="ossq.py --output db")
    parser.add_argument("--scanned", type=Path, required=True)
    parser.add_argument(
        "--component", default="ned", choices=COMPONENTS,
        help="Which NED figure to compare (default: overall).",
    )
    args = parser.parse_args()

    synthetic = read_scores(args.synthetic)
    scanned = read_scores(args.scanned)
    print(describe(pair_up(synthetic, scanned, args.component), args.component))


if __name__ == "__main__":
    main()
