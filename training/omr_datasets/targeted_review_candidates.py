"""Which specific pages are worth a human's manual review, not which scores.

The user's own framing: manual annotation is fine, even expected, but should go
where there's real evidence the auto-detector failed on a *specific* page - a score
whose other pages match its Lieder ground truth well (confirms we found the right
source and detection is basically working there) but one page doesn't is a much
stronger, more targeted signal than a low-scoring score in general, which could
just as easily mean a bad source match, wrong page alignment, or bar-line-detector
noise across the whole piece. The user's own hypothesis: this is most likely to be
the first or last page of a piece (piano-only intro/outro systems `_group_by_
geometry` doesn't always recover) - this reads `compare_bar_counts.py`'s
`--rows-out` data to check that directly, not just assume it.

A score is only eligible at all if most of its systems agree (`--min-score-exact-
fraction`, default 0.7) - the same one bad page inside an otherwise-good score
this module exists to surface, not a score that's wrong everywhere for some other
reason entirely.
"""

# flake8: noqa: T201

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rows(rows_path: Path) -> list[dict]:
    return json.loads(rows_path.read_text(encoding="utf-8"))


def group_by_score(rows: list[dict]) -> dict[str, list[dict]]:
    by_score: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_score[row["score_id"]].append(row)
    return by_score


def targeted_candidates(
    rows: list[dict], min_score_exact_fraction: float = 0.7
) -> list[dict]:
    """One row per *mismatching system*, restricted to scores whose other systems
    mostly agree - the targeted, "this specific page is probably wrong" list, not
    a blanket "this whole score scored low" list.
    """
    candidates = []
    for score_id, score_rows in group_by_score(rows).items():
        exact = sum(1 for row in score_rows if row["detected"] == row["ground_truth"])
        total = len(score_rows)
        if total == 0 or exact / total < min_score_exact_fraction:
            continue
        for row in score_rows:
            if row["detected"] != row["ground_truth"]:
                candidates.append({**row, "score_exact_fraction": exact / total})
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--rows", type=Path, required=True, help="compare_bar_counts.py's --rows-out file."
    )
    parser.add_argument(
        "--min-score-exact-fraction", type=float, default=0.7,
        help="Only surface mismatches from scores whose other systems agree at "
        "least this often - below that, the whole score is suspect, not one page.",
    )
    parser.add_argument("--out", type=Path, help="Write the candidate list here too, as JSON.")
    args = parser.parse_args()

    rows = load_rows(args.rows)
    candidates = targeted_candidates(rows, args.min_score_exact_fraction)

    print(f"{len(candidates)} targeted review candidate(s) "
          f"(from scores with >= {args.min_score_exact_fraction:.0%} of systems agreeing)")

    first_or_last = sum(1 for c in candidates if c["is_first_page"] or c["is_last_page"])
    print(f"{first_or_last}/{len(candidates)} are on a piece's first or last page")

    candidates.sort(key=lambda c: (c["score_id"], c["page_index"], c["system_index"]))
    for c in candidates:
        position = "first page" if c["is_first_page"] else "last page" if c["is_last_page"] else "middle"
        print(
            f"  {c['score_id']} / {c['page_image']} / system {c['system_index']} "
            f"({position}): detected {c['detected']}, ground truth {c['ground_truth']} "
            f"(score overall: {c['score_exact_fraction']:.0%} systems exact)"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(candidates, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
