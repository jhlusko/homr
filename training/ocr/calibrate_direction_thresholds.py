"""Choose per-class box-confidence thresholds on selection pages only.

B4's full-model checkpoints raised real OSSQ direction recall but predict far more boxes
than the frozen B3 test caps allow (2.2x the parent's direction boxes, 1.7x `e4`'s
Dynamic boxes on the selection pages). The owner accepted, on 2026-09-25, fixing an
operating point before the single test read: for each capped class, the lowest box
confidence threshold at which the checkpoint predicts no more boxes on the *selection*
pages than the reference model predicts on those same pages - the parent for
`DirectionText` (OSSQ and Lieder), released `e4` for `Dynamic` (OSSQ). One threshold per
class covers every corpus, so the stricter corpus sets it.

The caps are the frozen B3 rule, the pages are the frozen selection role, and test pages
are never read, so this does not tune on test. Recall at the calibrated operating point is
what `select_direction_epoch` then ranks: the output has the same shape as an
`eval_folded_directions` selection report, plus `thresholds`, `caps` and the
uncalibrated counts.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from training.ocr.detector_inference import load_model
from training.ocr.eval_folded_directions import (
    DIRECTION_CLASSES,
    apply_thresholds,
    folded,
    load_rows,
    predict_pages,
    score_predictions,
)

CORPORA = {"ossq": "ossq_boxes", "lieder": "lieder_boxes", "lyrics": "lieder_lyrics"}

#: (class, corpus, reference report) - the frozen B3 prediction-count guard.
CAPPED = (
    ("DirectionText", "ossq_boxes", "parent"),
    ("DirectionText", "lieder_boxes", "parent"),
    ("Dynamic", "ossq_boxes", "e4"),
)


def reference_caps(parent: dict, e4: dict) -> dict[str, dict[str, int]]:
    references = {"parent": parent, "e4": e4}
    caps: dict[str, dict[str, int]] = {}
    for label, corpus, name in CAPPED:
        counts = references[name]["corpora"][corpus]["selection"]["folded"][label]
        caps.setdefault(label, {})[corpus] = counts["predicted"]
    return caps


def threshold_for_cap(confidences: list[float], cap: int) -> float:
    """Smallest threshold keeping at most `cap` boxes (boxes kept when conf >= threshold).

    Ties at the cut are all dropped, so the kept count never exceeds the cap.
    """
    if len(confidences) <= cap:
        return 0.0
    ordered = np.sort(np.asarray(confidences, dtype=np.float64))[::-1]
    return float(np.nextafter(ordered[cap], np.inf))


def choose_thresholds(
    predictions: dict[str, list[tuple[dict, list, list]]], caps: dict[str, dict[str, int]]
) -> dict[str, float]:
    thresholds = {}
    for label, per_corpus in caps.items():
        needed = []
        for corpus, cap in per_corpus.items():
            confidences = [
                box.confidence
                for _row, _truth, predicted in predictions[corpus]
                for box in predicted
                if folded(box.label) == label
            ]
            needed.append(threshold_for_cap(confidences, cap))
        thresholds[label] = max(needed)
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pages", type=Path, nargs="+", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True, help="B3 parent baseline report")
    parser.add_argument("--e4", type=Path, required=True, help="B3 released e4 baseline report")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest_bytes = args.split_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    parent = json.loads(args.parent.read_text())
    e4 = json.loads(args.e4.read_text())
    if not parent["split_manifest_sha256"] == e4["split_manifest_sha256"] == manifest_sha:
        raise ValueError("baseline reports and split manifest disagree")
    caps = reference_caps(parent, e4)

    model = load_model(args.weights, args.device)
    predictions = {}
    for pages_path in args.pages:
        name = pages_path.name.removesuffix(".pages.jsonl").removeprefix("run1-")
        corpus = CORPORA.get(name, name)
        entry = manifest["corpora"][corpus]
        if hashlib.sha256(pages_path.read_bytes()).hexdigest() != entry["source_pages_sha256"]:
            raise ValueError(f"page-row digest mismatch: {pages_path}")
        role = entry["roles"]["selection"]
        rows = load_rows(pages_path, set(role["scores"]))
        if len(rows) != role["pages"]:
            raise ValueError(f"page count mismatch: {corpus}/selection")
        predictions[corpus] = predict_pages(model, rows, args.device, f"{corpus}/selection")
    if {corpus for _label, corpus, _name in CAPPED} - set(predictions):
        raise ValueError("every capped corpus must be scored")

    thresholds = choose_thresholds(predictions, caps)
    report = {
        "weights": str(args.weights),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "class_order": model.detector_class_order,
        "direction_classes": sorted(DIRECTION_CLASSES),
        "iou_threshold": 0.5,
        "split_manifest_sha256": manifest_sha,
        "calibration": "selection-only per-class confidence thresholds at the B3 prediction caps",
        "caps": caps,
        "thresholds": thresholds,
        "corpora": {},
        "uncalibrated": {},
    }
    for corpus, pages in predictions.items():
        report["uncalibrated"][corpus] = score_predictions(pages)["folded"]
        calibrated = [(row, truth, apply_thresholds(boxes, thresholds)) for row, truth, boxes in pages]
        report["corpora"][corpus] = {"selection": score_predictions(calibrated, include_page_rows=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        corpus: {role: value["folded"] for role, value in roles.items()}
        for corpus, roles in report["corpora"].items()
    }
    print(json.dumps({"thresholds": thresholds, "caps": caps, "calibrated": summary}, indent=2))


if __name__ == "__main__":
    main()
