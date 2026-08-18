"""
Detect a syllable's box on a scan, then read it - the gap 27.45 opened and 27.68 first
measured the data half of.

Every number the recogniser has produced so far (`train_recognizer.py`) used a crop the
detector never had to find - MuseScore's own ground-truth box. This is the first place
those two models actually meet: the detector's own predicted Lyrics box, matched to a
ground-truth box by IoU (`detector_box_eval.py`'s matching, reused rather than
reimplemented), determines what gets cropped and handed to the recogniser. The oracle
number (same matched syllables, ground-truth box instead) is measured alongside so a drop
is attributable to localisation rather than reading - the same reason 27.86 kept detection
and recognition as separate modules instead of one end-to-end script from the start.

Ground truth here comes straight from `*.boxes.json`, which carries a lyric's text next to
its box - `lyric_crops/*.jsonl` only has pre-cut crops with no page coordinates, so it
cannot answer "what did the detector find," only "can the recogniser read a syllable
someone else already located."
"""

# flake8: noqa: T201

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from training.architecture.ocr.crnn import CRNN, IMAGE_HEIGHT, Alphabet
from training.ocr.detector_box_eval import iou
from training.ocr.detector_inference import PredictedBox, load_model, predict_boxes
from training.omr_datasets.lyric_crops import MARGIN
from training.ocr.recognizer_data import scaled_width
from training.ocr.train_recognizer import Accuracy, edit_distance


@dataclass(frozen=True)
class LyricBox:
    text: str
    left: int
    top: int
    right: int
    bottom: int


def ground_truth_lyrics(boxes_json: Path) -> list[LyricBox]:
    record = json.loads(boxes_json.read_text(encoding="utf-8"))
    return [
        LyricBox(lyric["text"], lyric["left"], lyric["top"], lyric["right"], lyric["bottom"])
        for lyric in record.get("lyrics", [])
    ]


def boxes_json_for(image_path: Path) -> Path:
    """`.../<stem>/<stem>-<page>.png` -> `.../<stem>/<stem>.boxes.json` - one boxes.json
    covers every page image in its folder, named after the folder rather than the page.
    """
    return image_path.parent / f"{image_path.parent.name}.boxes.json"


def match_with_text(
    predicted: list[PredictedBox], ground_truth: list[LyricBox], iou_threshold: float = 0.5
) -> list[tuple[PredictedBox, LyricBox]]:
    """Greedy one-to-one IoU matching, same rule as `detector_box_eval.match_one_page`,
    kept separate because that function only ever needed counts, not the text pairing."""
    candidates = []
    for p_index, pred in enumerate(predicted):
        if pred.label != "Lyrics":
            continue
        for g_index, gt in enumerate(ground_truth):
            score = iou(
                (pred.left, pred.top, pred.right, pred.bottom),
                (gt.left, gt.top, gt.right, gt.bottom),
            )
            if score >= iou_threshold:
                candidates.append((score, p_index, g_index))
    candidates.sort(reverse=True)

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    pairs = []
    for score, p_index, g_index in candidates:
        if p_index in used_pred or g_index in used_gt:
            continue
        used_pred.add(p_index)
        used_gt.add(g_index)
        pairs.append((predicted[p_index], ground_truth[g_index]))
    return pairs


def crop_for_recognizer(image: np.ndarray, box: tuple[int, int, int, int], height: int) -> torch.Tensor:
    """Same preprocessing the training crops went through: `lyric_crops.py`'s `MARGIN` of
    air kept around the tight box (a recogniser reads better with room for a hyphen or a
    descender than a box cut exactly to the ink - cropping tight here was the first version
    of this function and it made both the oracle and the detected-box numbers implausibly
    low, well below the recogniser's own training-time accuracy, before this fix), then
    resized to `height` keeping aspect ratio and normalised to [0, 1]."""
    left, top, right, bottom = box
    height_bound, width_bound = image.shape[:2]
    left = max(0, left - MARGIN)
    top = max(0, top - MARGIN)
    right = min(width_bound, right + MARGIN)
    bottom = min(height_bound, bottom + MARGIN)
    crop = image[top:bottom, left:right]
    if crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
        crop = np.full((height, 8), 255, dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    width = scaled_width(gray.shape[1], gray.shape[0], height)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)


@torch.no_grad()
def read_crop(model: CRNN, alphabet: Alphabet, crop: torch.Tensor, device: str) -> str:
    tensor = crop.unsqueeze(0).to(device)
    logits = model(tensor)
    best = logits.argmax(dim=-1).permute(1, 0).cpu()[0]
    frames = model.frame_count(tensor.shape[-1])
    return alphabet.decode(best[:frames].tolist())


def evaluate(
    detector_weights: Path,
    recognizer_weights: Path,
    index: Path,
    device: str,
    iou_threshold: float = 0.5,
) -> dict:
    detector = load_model(detector_weights, device)
    checkpoint = torch.load(recognizer_weights, map_location=device)
    alphabet = Alphabet(checkpoint["alphabet"])
    recognizer = CRNN(len(alphabet), image_height=IMAGE_HEIGHT).to(device)
    recognizer.load_state_dict(checkpoint["model"])
    recognizer.eval()

    detected = Accuracy()  # recognizer reading the detector's own predicted box
    oracle = Accuracy()  # same matched syllables, ground-truth box instead
    match_counts: dict[str, int] = collections.defaultdict(int)

    images = [line.split(",")[0] for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    for page_index, image_path_str in enumerate(images):
        if page_index % 20 == 0:
            print(f"  page {page_index}/{len(images)}")
        image_path = Path(image_path_str)
        boxes_json = boxes_json_for(image_path)
        if not boxes_json.is_file():
            continue
        ground_truth = ground_truth_lyrics(boxes_json)
        match_counts["ground_truth"] += len(ground_truth)
        if not ground_truth:
            continue

        page = cv2.imread(str(image_path))
        if page is None:
            continue
        predicted = predict_boxes(detector, image_path, device)
        pairs = match_with_text(predicted, ground_truth, iou_threshold)
        match_counts["matched"] += len(pairs)

        for pred_box, gt in pairs:
            detected_crop = crop_for_recognizer(
                page, (pred_box.left, pred_box.top, pred_box.right, pred_box.bottom), IMAGE_HEIGHT
            )
            oracle_crop = crop_for_recognizer(
                page, (gt.left, gt.top, gt.right, gt.bottom), IMAGE_HEIGHT
            )
            detected.observe(gt.text, read_crop(recognizer, alphabet, detected_crop, device))
            oracle.observe(gt.text, read_crop(recognizer, alphabet, oracle_crop, device))

    return {"detected": detected, "oracle": oracle, "counts": dict(match_counts)}


def describe(result: dict) -> str:
    counts = result["counts"]
    lines = [
        f"{counts.get('ground_truth', 0):,} ground-truth lyric boxes, "
        f"{counts.get('matched', 0):,} matched to a detected box (IoU-based, same as "
        "detector_box_eval)",
        "",
        result["detected"].describe("read from the detector's own box  "),
        result["oracle"].describe("read from the ground-truth box    "),
    ]
    if result["detected"].examples:
        lines.append("")
        lines.append(
            "misreads from the detector's box: "
            + ", ".join(f"{t!r}->{p!r}" for t, p in result["detected"].examples[:6])
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--recognizer-weights", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="detector_split valid_index.txt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = evaluate(
        args.detector_weights, args.recognizer_weights, args.index, args.device, args.iou_threshold
    )
    print(describe(result))


if __name__ == "__main__":
    main()
