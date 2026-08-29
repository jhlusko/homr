"""Turn a checkpoint-diff review export into an answer about where a delta comes from.

An aggregate says one checkpoint is ahead. It cannot say whether that is better reading
of the page, a token the older checkpoint has no way to emit, or a reference that is
simply wrong - and those three call for completely different responses. The review set's
verdicts separate them; this adds up what each is worth.

**These sums do not extrapolate to the benchmark.** The set is the largest disagreements,
not a random sample, so the totals here describe the reviewed staves and nothing else.
Read them as "of the movement a reviewer actually looked at, this much was vocabulary" -
not as a correction to the benchmark figure.
"""

# flake8: noqa: T201

import argparse
import json
from collections import defaultdict
from pathlib import Path

#: Verdicts that do not describe one checkpoint reading the page better than the other.
#: Kept explicit rather than inferred from the name: a category quietly falling into the
#: wrong bucket is the one way this summary can mislead.
NOT_A_READING_DIFFERENCE = {"vocab-only", "ref-wrong", "unclear", "same"}


def summarise(export: dict) -> dict:
    rows = export.get("reviewed", [])
    judged = [r for r in rows if r.get("verdict")]
    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for row in judged:
        by_verdict[row["verdict"]].append(row)

    reading = [r for r in judged if r["verdict"] not in NOT_A_READING_DIFFERENCE]
    vocabulary = by_verdict.get("vocab-only", [])
    reference = by_verdict.get("ref-wrong", [])

    def total(items: list[dict]) -> float:
        return sum(float(i.get("delta") or 0.0) for i in items)

    return {
        "set": export.get("set"),
        "items": len(rows),
        "judged": len(judged),
        "unjudged": len(rows) - len(judged),
        "by_verdict": {
            verdict: {
                "n": len(items),
                "sum_delta": round(total(items), 4),
                "mean_delta": round(total(items) / len(items), 4) if items else 0.0,
            }
            for verdict, items in sorted(by_verdict.items(), key=lambda kv: -len(kv[1]))
        },
        "movement": {
            "judged_total": round(total(judged), 4),
            "reading_difference": round(total(reading), 4),
            "vocabulary_only": round(total(vocabulary), 4),
            "reference_wrong": round(total(reference), 4),
        },
        "notes": [
            {"id": r["id"], "verdict": r["verdict"], "notes": r["notes"]}
            for r in judged
            if r.get("notes")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, required=True, help="the exported .json")
    parser.add_argument("--out", type=Path, help="write the summary as JSON as well")
    args = parser.parse_args()

    result = summarise(json.loads(args.review.read_text(encoding="utf-8")))

    print(f"{result['set']}: {result['judged']} of {result['items']} judged", end="")
    print(f", {result['unjudged']} left" if result["unjudged"] else "")
    print()
    print(f"{'verdict':<14}{'n':>5}{'sum delta':>12}{'mean':>10}")
    for verdict, stats in result["by_verdict"].items():
        print(f"{verdict:<14}{stats['n']:>5}{stats['sum_delta']:>+12.3f}{stats['mean_delta']:>+10.3f}")

    movement = result["movement"]
    print()
    print(f"movement over judged staves: {movement['judged_total']:+.3f}")
    print(f"  a genuine reading difference: {movement['reading_difference']:+.3f}")
    print(f"  vocabulary only:              {movement['vocabulary_only']:+.3f}")
    print(f"  reference was wrong:          {movement['reference_wrong']:+.3f}")
    print()
    print("This set is the largest disagreements, not a sample - these totals describe")
    print("the reviewed staves and do not extrapolate to the benchmark.")

    if result["notes"]:
        print(f"\n{len(result['notes'])} item(s) with reviewer notes:")
        for note in result["notes"]:
            print(f"  {note['id']} [{note['verdict']}] {note['notes']}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
