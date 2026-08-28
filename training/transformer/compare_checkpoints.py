"""Compare checkpoints on one benchmark, as a paired test over the same staves.

The point of this file is that the previous write-up reported roughly +3% for what was
a substantial regression. That happened because every number in it came from labels
this project had itself rebuilt, scored ad hoc, with no paired test and no independent
corpus. Two properties are therefore deliberate here:

**Paired, not two independent means.** Checkpoints are scored on identical staves, so
the comparison is a per-staff difference. Reporting two aggregate accuracies hides how
a difference is distributed - a 1pp gap is a very different fact when it is 8 staves
collapsing than when it is every staff drifting - and `wins`/`losses`/`ties` below say
which. The bootstrap is over staves and preserves the pairing.

**Pooled positions, per branch, and macro.** `overall` pools every branch's positions,
so long staves and dense branches count for more; `macro` averages the six branch
accuracies. They answer different questions and can disagree, so both are printed
rather than one being chosen silently.

Accuracy itself uses the padding contract from `base_predictions.py`: reference and
prediction are already padded to a common width there, so a length disagreement is
counted as a miss on both sides rather than being normalised away.
"""

# flake8: noqa: T201

import argparse
import json
import random
from pathlib import Path

BRANCHES = ("pitch", "rhythm", "lift", "articulation", "slur", "position")


def load(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """staff id -> branch -> (correct, total)."""
    out: dict[str, dict[str, tuple[int, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        per_branch = {}
        for branch in BRANCHES:
            want = row.get(f"{branch}_reference")
            got = row.get(f"{branch}_predicted")
            if want is None or got is None:
                continue
            correct = sum(1 for a, b in zip(want, got, strict=False) if a == b)
            per_branch[branch] = (correct, max(len(want), len(got)))
        out[row["tokens"]] = per_branch
    return out


def accuracy(staves: list[dict[str, tuple[int, int]]]) -> dict[str, float]:
    result = {}
    pooled_correct = pooled_total = 0
    for branch in BRANCHES:
        correct = sum(s[branch][0] for s in staves if branch in s)
        total = sum(s[branch][1] for s in staves if branch in s)
        result[branch] = correct / total if total else float("nan")
        pooled_correct += correct
        pooled_total += total
    result["overall"] = pooled_correct / pooled_total if pooled_total else float("nan")
    branch_values = [result[b] for b in BRANCHES if result[b] == result[b]]
    result["macro"] = sum(branch_values) / len(branch_values) if branch_values else float("nan")
    return result


def staff_overall(per_branch: dict[str, tuple[int, int]]) -> float:
    correct = sum(c for c, _ in per_branch.values())
    total = sum(t for _, t in per_branch.values())
    return correct / total if total else float("nan")


def bootstrap_delta(
    a: list[dict], b: list[dict], rounds: int, seed: int
) -> tuple[float, float]:
    """95% interval on the paired difference b - a, resampling staves together."""
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(
            accuracy([b[i] for i in idx])["overall"] - accuracy([a[i] for i in idx])["overall"]
        )
    deltas.sort()
    return deltas[int(0.025 * rounds)], deltas[int(0.975 * rounds)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="benchmark name, for the report")
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=PATH",
        help="a scored .jsonl; repeat. The first is the baseline every other is paired against.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        if not path:
            parser.error(f"--run wants LABEL=PATH, got {spec!r}")
        runs.append((label, load(Path(path))))

    # Only staves every run scored. A checkpoint that failed on some crop would
    # otherwise be compared on an easier subset than the one it is measured against.
    shared = set(runs[0][1])
    for _, rows in runs[1:]:
        shared &= set(rows)
    ids = sorted(shared)
    dropped = {label: len(rows) - len(ids) for label, rows in runs}
    print(f"{args.name}: {len(ids):,} staves scored by all {len(runs)} runs")
    for label, count in dropped.items():
        if count:
            print(f"  note: {count} staves dropped from {label} (not scored by every run)")

    report = {"benchmark": args.name, "staves": len(ids), "runs": {}}
    baseline_label, baseline_rows = runs[0]
    baseline = [baseline_rows[i] for i in ids]
    base_acc = accuracy(baseline)

    header = f"{'run':<10}" + "".join(f"{b[:9]:>10}" for b in BRANCHES) + f"{'overall':>10}{'macro':>10}"
    print(header)
    for label, rows in runs:
        staves = [rows[i] for i in ids]
        acc = accuracy(staves)
        print(
            f"{label:<10}"
            + "".join(f"{acc[b] * 100:>9.2f} " for b in BRANCHES)
            + f"{acc['overall'] * 100:>9.2f} {acc['macro'] * 100:>9.2f}"
        )
        entry = {b: acc[b] for b in (*BRANCHES, "overall", "macro")}
        if label != baseline_label:
            entry["delta_overall"] = acc["overall"] - base_acc["overall"]
            low, high = bootstrap_delta(baseline, staves, args.bootstrap, args.seed)
            entry["delta_ci95"] = [low, high]
            wins = losses = ties = 0
            for i in ids:
                d = staff_overall(rows[i]) - staff_overall(baseline_rows[i])
                if d > 1e-12:
                    wins += 1
                elif d < -1e-12:
                    losses += 1
                else:
                    ties += 1
            entry.update(wins=wins, losses=losses, ties=ties)
        report["runs"][label] = entry

    print()
    for label, entry in report["runs"].items():
        if "delta_overall" not in entry:
            continue
        low, high = entry["delta_ci95"]
        verdict = "significant" if low > 0 or high < 0 else "NOT significant"
        print(
            f"{label} vs {baseline_label}: {entry['delta_overall'] * 100:+.2f}pp "
            f"(95% CI {low * 100:+.2f} to {high * 100:+.2f}, {verdict}) "
            f"- staves better {entry['wins']}, worse {entry['losses']}, tied {entry['ties']}"
        )

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
