"""Score a detector's direction-text classes as one class on saved real-page indexes.

The input page JSONL files come from the 2026-09-19 real-page evaluator. Each row names
the exact image and annotation used by that run. Re-running inference is necessary:
per-class counts cannot recover a Tempo prediction overlapping StaffText ground truth.
The unmerged counts are scored in the same pass to check against the saved report.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from training.ocr.detector_box_eval import Counts, match_one_page
from training.ocr.detector_data import boxes_of
from training.ocr.detector_inference import load_model, predict_boxes
from training.ocr.detector_masks import CLASS_ORDER, canonical_label


DIRECTION_CLASSES = frozenset({"Tempo", "StaffText", "Expression", "SystemText"})


def folded(label: str) -> str:
    return "DirectionText" if label in DIRECTION_CLASSES else label


def add_counts(total: dict[str, Counts], page: dict[str, Counts]) -> None:
    for label, counts in page.items():
        total[label].matched += counts.matched
        total[label].predicted += counts.predicted
        total[label].ground_truth += counts.ground_truth


def serialize(total: dict[str, Counts]) -> dict[str, dict]:
    return {
        label: {
            "matched": counts.matched,
            "predicted": counts.predicted,
            "ground_truth": counts.ground_truth,
            "recall": counts.matched / counts.ground_truth if counts.ground_truth else None,
            "precision_lower_bound": (
                counts.matched / counts.predicted if counts.predicted else None
            ),
        }
        for label, counts in sorted(total.items())
    }


def score_pages(model, pages_path: Path, device: str, limit: int = 0) -> dict:
    plain: dict[str, Counts] = defaultdict(Counts)
    merged: dict[str, Counts] = defaultdict(Counts)
    rows = [json.loads(line) for line in pages_path.read_text().splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    for index, row in enumerate(rows, 1):
        image = row["image"]
        annotation = json.loads(Path(row["annotation"]).read_text())
        ground_truth = [
            replace(box, label=canonical_label(box.label))
            for box in boxes_of(annotation, image)
            if canonical_label(box.label) in CLASS_ORDER
        ]
        predicted = predict_boxes(model, Path(image), device)
        add_counts(plain, match_one_page(predicted, ground_truth, 0.5))
        add_counts(
            merged,
            match_one_page(
                [replace(box, label=folded(box.label)) for box in predicted],
                [replace(box, label=folded(box.label)) for box in ground_truth],
                0.5,
            ),
        )
        if index % 20 == 0 or index == len(rows):
            print(f"{pages_path.stem}: {index}/{len(rows)} pages", flush=True)
    return {"pages": len(rows), "per_class": serialize(plain), "folded": serialize(merged)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pages", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; omit for report")
    args = parser.parse_args()
    model = load_model(args.weights, args.device)
    report = {
        "weights": str(args.weights),
        "direction_classes": sorted(DIRECTION_CLASSES),
        "iou_threshold": 0.5,
        "corpora": {},
    }
    for pages_path in args.pages:
        name = pages_path.name.removesuffix(".pages.jsonl").removeprefix("run1-")
        report["corpora"][name] = score_pages(model, pages_path, args.device, args.limit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
