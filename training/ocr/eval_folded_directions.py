"""Score a detector's direction-text classes as one class on saved real-page indexes.

The input page JSONL files come from the 2026-09-19 real-page evaluator. Each row names
the exact image and annotation used by that run. Re-running inference is necessary:
per-class counts cannot recover a Tempo prediction overlapping StaffText ground truth.
The unmerged counts are scored in the same pass to check against the saved report.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

from training.ocr.detector_box_eval import Counts, match_one_page
from training.ocr.detector_data import boxes_of
from training.ocr.detector_inference import load_model, predict_boxes


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


def reference_label(label: str, model_order: tuple[str, ...]) -> str:
    if "DirectionText" in model_order:
        return folded(label)
    return "StaffText" if label == "SystemText" else label


def score_pages(
    model,
    pages_path: Path,
    device: str,
    limit: int = 0,
    scores: set[str] | None = None,
    include_page_rows: bool = False,
) -> dict:
    plain: dict[str, Counts] = defaultdict(Counts)
    merged: dict[str, Counts] = defaultdict(Counts)
    rows = [json.loads(line) for line in pages_path.read_text().splitlines() if line.strip()]
    if scores is not None:
        rows = [row for row in rows if row["score"] in scores]
    if limit:
        rows = rows[:limit]
    page_rows = []
    model_order = model.detector_class_order
    for index, row in enumerate(rows, 1):
        image = row["image"]
        annotation = json.loads(Path(row["annotation"]).read_text())
        ground_truth = [
            replace(box, label=reference_label(box.label, model_order))
            for box in boxes_of(annotation, image)
            if reference_label(box.label, model_order) in model_order
        ]
        predicted = predict_boxes(model, Path(image), device)
        plain_page = match_one_page(predicted, ground_truth, 0.5)
        folded_page = match_one_page(
            [replace(box, label=folded(box.label)) for box in predicted],
            [replace(box, label=folded(box.label)) for box in ground_truth],
            0.5,
        )
        add_counts(plain, plain_page)
        add_counts(merged, folded_page)
        if include_page_rows:
            page_rows.append(
                {
                    "score": row["score"],
                    "image": image,
                    "per_class": serialize(plain_page),
                    "folded": serialize(folded_page),
                }
            )
        if index % 20 == 0 or index == len(rows):
            print(f"{pages_path.stem}: {index}/{len(rows)} pages", flush=True)
    result = {"pages": len(rows), "per_class": serialize(plain), "folded": serialize(merged)}
    if include_page_rows:
        result["page_rows"] = page_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pages", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--classes", help="Comma-separated order for older weights without a sidecar"
    )
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; omit for report")
    parser.add_argument("--split-manifest", type=Path, help="Frozen score-level selection/test split")
    parser.add_argument(
        "--role", choices=("selection", "test", "both"), default="both",
        help="When using a split manifest, avoid reading test pages during epoch selection",
    )
    args = parser.parse_args()
    order = tuple(args.classes.split(",")) if args.classes else None
    model = load_model(args.weights, args.device, order)
    manifest_bytes = args.split_manifest.read_bytes() if args.split_manifest else None
    manifest = json.loads(manifest_bytes) if manifest_bytes else None
    if args.role != "both" and manifest is None:
        parser.error("--role requires --split-manifest")
    report = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "class_order": model.detector_class_order,
        "direction_classes": sorted(DIRECTION_CLASSES),
        "iou_threshold": 0.5,
        "corpora": {},
    }
    if manifest_bytes:
        report["split_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    for pages_path in args.pages:
        name = pages_path.name.removesuffix(".pages.jsonl").removeprefix("run1-")
        corpus = {"ossq": "ossq_boxes", "lieder": "lieder_boxes", "lyrics": "lieder_lyrics"}.get(name, name)
        if manifest:
            entry = manifest["corpora"][corpus]
            if hashlib.sha256(pages_path.read_bytes()).hexdigest() != entry["source_pages_sha256"]:
                raise ValueError(f"page-row digest mismatch: {pages_path}")
            report["corpora"][corpus] = {}
            roles = ("selection", "test") if args.role == "both" else (args.role,)
            for role in roles:
                role_info = entry["roles"][role]
                result = score_pages(
                    model, pages_path, args.device, args.limit,
                    set(role_info["scores"]), include_page_rows=True,
                )
                if not args.limit and result["pages"] != role_info["pages"]:
                    raise ValueError(f"page count mismatch: {corpus}/{role}")
                report["corpora"][corpus][role] = result
        else:
            report["corpora"][name] = score_pages(model, pages_path, args.device, args.limit)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
