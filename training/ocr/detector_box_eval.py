"""
Does the detector find the boxes, not just the pixels?

27.86 measured per-pixel IoU on the training patches; `detector_inference.py` runs the
model over a whole page and recovers boxes the way inference actually has to. This is the
localisation number those boxes were built for: match predicted boxes against
`detector_data.py`'s ground truth by IoU and class, per page, and report precision/recall/F1
- the number that decides whether the detector is ready to feed the recognizer crops
(27.86's stated next step), separate from whether the recognizer can read a crop once
handed one.

Matching is greedy by descending IoU, one-to-one: a ground-truth box already claimed by a
higher-IoU prediction cannot also match a lower one, and vice versa - the usual detection
matching rule, needed because two adjacent syllables can each have a plausible-looking
overlap with the same predicted box.
"""

# flake8: noqa: T201

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from training.ocr.detector_data import Box, collect
from training.ocr.detector_inference import load_model, predict_boxes
from training.ocr.detector_masks import CLASS_INDEX, canonical_label
from training.ocr.detector_patches import read_index


@dataclass
class Counts:
    matched: int = 0
    predicted: int = 0
    ground_truth: int = 0

    @property
    def precision(self) -> float:
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.matched / self.ground_truth if self.ground_truth else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_one_page(
    predicted: list, ground_truth: list[Box], iou_threshold: float
) -> dict[str, Counts]:
    """Greedy one-to-one IoU matching, scored per class."""
    by_class: dict[str, Counts] = collections.defaultdict(Counts)
    for gt in ground_truth:
        by_class[gt.label].ground_truth += 1
    for pred in predicted:
        by_class[pred.label].predicted += 1

    candidates = []
    for p_index, pred in enumerate(predicted):
        for g_index, gt in enumerate(ground_truth):
            if pred.label != gt.label:
                continue
            score = iou((pred.left, pred.top, pred.right, pred.bottom), (gt.left, gt.top, gt.right, gt.bottom))
            if score >= iou_threshold:
                candidates.append((score, p_index, g_index))
    candidates.sort(reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for score, p_index, g_index in candidates:
        if p_index in used_pred or g_index in used_gt:
            continue
        used_pred.add(p_index)
        used_gt.add(g_index)
        by_class[predicted[p_index].label].matched += 1
    return by_class


def evaluate(
    weights: Path, boxes_dir: Path, index: Path, device: str, iou_threshold: float = 0.5
) -> dict[str, Counts]:
    model = load_model(weights, device)
    # `collect` reads detector_data.DETECTION_CLASSES (11 classes); the model was only
    # ever trained on detector_masks.CLASS_ORDER (6) - Text, InstrumentName,
    # RehearsalMark and Harmony were never rasterized into a training mask, so scoring
    # them here would count boxes the detector was never asked to find as missed recall.
    # `canonical_label` folds SystemText into StaffText (27.92) before that filter, so a
    # ground-truth SystemText box is scored as a StaffText box, matching what the masks
    # (and therefore the model's own labels) were built from.
    ground_truth_boxes = [
        Box(box.image, canonical_label(box.label), box.left, box.top, box.right, box.bottom)
        for box in collect(boxes_dir)
        if canonical_label(box.label) in CLASS_INDEX
    ]
    by_image: dict[str, list[Box]] = collections.defaultdict(list)
    for box in ground_truth_boxes:
        by_image[box.image].append(box)

    totals: dict[str, Counts] = collections.defaultdict(Counts)
    # Via `read_index` rather than splitting here: OSSQ files pages by composer, so
    # image paths contain commas ("Haydn,_Joseph/..."), and taking the first
    # comma-separated field silently truncates the path to a directory that does not
    # exist. One parser for the index format, in one place.
    images = [sample.image for sample in read_index(index)]
    for page_index, image_path in enumerate(images):
        if page_index % 20 == 0:
            print(f"  page {page_index}/{len(images)}")
        predicted = predict_boxes(model, Path(image_path), device)
        page_counts = match_one_page(predicted, by_image.get(image_path, []), iou_threshold)
        for label, counts in page_counts.items():
            totals[label].matched += counts.matched
            totals[label].predicted += counts.predicted
            totals[label].ground_truth += counts.ground_truth
    return totals


#: The classes this project actually needs the detector to get right (user decision,
#: 2026-08-25). Lyrics and Dynamic are 93% of all ground-truth boxes in this corpus.
#: MeasureNumber is derivable rather than detected; Tempo is rare enough to fix by hand
#: (roughly once per movement); Fingering is not wanted; StaffText and Expression are
#: wanted but explicitly expendable if giving them up buys Lyrics and Dynamic.
PRIORITY_CLASSES = ("Lyrics", "Dynamic")


def describe(totals: dict[str, Counts], priority: tuple[str, ...] = PRIORITY_CLASSES) -> str:
    """Per-class rows, an all-class total, and a total over `priority` alone.

    Both totals are reported because they answer different questions and disagree
    sharply here. The all-class row is dominated by classes with a few dozen boxes and
    weak baselines - it read 1.2% for a model scoring 86% F1 on the classes that carry
    93% of the corpus, which made a usable model look like a catastrophe. The priority
    row is the one tied to a decision; the all-class row stays so the cost of ignoring
    those classes is visible rather than hidden.
    """
    lines = [f"{'class':<14} {'precision':>10} {'recall':>10} {'f1':>10} {'gt boxes':>10}"]
    overall = Counts()
    focus = Counts()
    for label in sorted(totals):
        counts = totals[label]
        for bucket in (overall,) + ((focus,) if label in priority else ()):
            bucket.matched += counts.matched
            bucket.predicted += counts.predicted
            bucket.ground_truth += counts.ground_truth
        lines.append(
            f"{label:<14} {counts.precision:>10.1%} {counts.recall:>10.1%} "
            f"{counts.f1:>10.1%} {counts.ground_truth:>10,}"
        )
    lines.append(
        f"{'overall':<14} {overall.precision:>10.1%} {overall.recall:>10.1%} "
        f"{overall.f1:>10.1%} {overall.ground_truth:>10,}"
    )
    lines.append(
        f"{'priority':<14} {focus.precision:>10.1%} {focus.recall:>10.1%} "
        f"{focus.f1:>10.1%} {focus.ground_truth:>10,}   ({', '.join(priority)})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--boxes", type=Path, required=True, help="A musescore_boxes out dir.")
    parser.add_argument("--index", type=Path, required=True, help="detector_split valid_index.txt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    totals = evaluate(args.weights, args.boxes, args.index, args.device, args.iou_threshold)
    print(describe(totals))
    if args.out:
        args.out.write_text(
            json.dumps({k: v.__dict__ for k, v in totals.items()}, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
