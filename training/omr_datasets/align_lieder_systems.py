"""Build whole-score scan-to-measure alignments from physical bar counts.

Unlike the historical extractor, this never zips scanned systems to rendered
MuseScore systems by ordinal position.  It aligns complete count sequences with
many-to-many moves and reports ambiguous/false-positive systems for quarantine.
"""

# flake8: noqa: T201

import argparse
import json
from collections import defaultdict
from pathlib import Path

from training.omr_datasets.system_count_alignment import (
    DEFAULT_MIN_MARGIN,
    align_system_counts,
)

DEFAULT_MIN_SYSTEM_WIDTH_FRACTION = 0.35


def build_alignment_document(
    rows: list[dict],
    ground_truth_dir: Path,
    *,
    max_group: int,
    min_margin: float,
    min_width_fraction: float = DEFAULT_MIN_SYSTEM_WIDTH_FRACTION,
    wanted: set[str] | None = None,
) -> dict:
    by_score: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        score_id = row["score_id"]
        if wanted is None or score_id in wanted:
            by_score[score_id].append(row)

    scores = {}
    for score_id in sorted(by_score):
        gt_path = ground_truth_dir / f"{score_id}.json"
        if not gt_path.exists():
            continue
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        source_counts = [count for page in ground_truth["pages"] for count in page]
        score_rows = by_score[score_id]
        # Narrow detections are page ornaments/illustrations, not score systems.
        # IMSLP183800's first page produced two such boxes and shifted every real
        # crop by two positions in the old flat zip.  A zero cannot enter an exact
        # match, so the global aligner must skip it without consuming score music.
        scan_counts = [
            0
            if float(row.get("system_width_fraction", 1.0)) < min_width_fraction
            else int(row["detected"])
            for row in score_rows
        ]
        report = align_system_counts(
            scan_counts,
            source_counts,
            max_group=max_group,
            min_margin=min_margin,
        )
        for item, row in zip(report["systems"], score_rows, strict=True):
            item["observed_measures"] = int(row["detected"])
            item["page_index"] = row["page_index"]
            item["page_image"] = row["page_image"]
            item["system_index"] = row["system_index"]
        scores[score_id] = report

    counts = defaultdict(int)
    for report in scores.values():
        for item in report["systems"]:
            counts[item["status"]] += 1
    return {
        "version": 1,
        "method": "whole-score exact physical-measure-count alignment",
        "model_predictions_used": False,
        "max_group": max_group,
        "min_margin": min_margin,
        "min_width_fraction": min_width_fraction,
        "summary": dict(sorted(counts.items())),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--score-ids", type=Path)
    parser.add_argument("--max-group", type=int, default=4)
    parser.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    parser.add_argument(
        "--min-width-fraction", type=float, default=DEFAULT_MIN_SYSTEM_WIDTH_FRACTION
    )
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text(encoding="utf-8"))
    wanted = None
    if args.score_ids:
        wanted = {line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()}
    document = build_alignment_document(
        rows,
        args.ground_truth,
        max_group=args.max_group,
        min_margin=args.min_margin,
        min_width_fraction=args.min_width_fraction,
        wanted=wanted,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    summary = document["summary"]
    print(f"{len(document['scores'])} scores aligned")
    print(", ".join(f"{key}: {value}" for key, value in summary.items()))
    print(f"model predictions used: {document['model_predictions_used']}")


if __name__ == "__main__":
    main()
