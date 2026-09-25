"""
Run the trained detector over a full-resolution page and recover boxes.

27.86 measured the detector's per-pixel accuracy on masks the training patches were drawn
from - it never ran the model over a *page*, tiled the way inference actually has to. This
is that step: slide a `PATCH_SIZE` window with 50% overlap (matching the convention
`detector_masks.py` already pointed at, from homr's own `inference_segnet.extract`) across
the full page, stitch per-pixel class probabilities by averaging the overlap rather than
picking one tile's answer at a seam, then recover boxes from the stitched mask with the
same `connectedComponentsWithStats` `detector_masks.rasterize`'s ground truth used - so a
predicted box and a ground-truth box are comparable by construction, not by convention.

This module stops at boxes. Whether a predicted box, cropped from the page and run through
the recognizer, reads the right syllable is the next question and a separate module's job -
keeping them separate means a bug in one is not disguised as a bug in the other.
"""

# flake8: noqa: T201

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from training.architecture.segmentation.model import CamVidModel
from homr.text_detector_classes import read_class_order
from training.ocr.detector_masks import CLASS_ORDER
from training.ocr.detector_patches import PATCH_SIZE, extract_patch

NUM_CLASSES = len(CLASS_ORDER) + 1
CLASS_NAMES = ("background", *CLASS_ORDER)
STEP = PATCH_SIZE // 2


@dataclass(frozen=True)
class PredictedBox:
    label: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


def load_model(
    weights: Path, device: str, class_order: tuple[str, ...] | None = None
) -> CamVidModel:
    order = read_class_order(weights, class_order)
    model = CamVidModel(
        arch="Unet",
        encoder_name="resnet18",
        in_channels=3,
        out_classes=len(order) + 1,
        skip_weights_download=True,
    )
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.detector_class_order = order
    model.to(device).eval()
    return model


def tile_origins(height: int, width: int, step: int = STEP) -> list[tuple[int, int]]:
    """Top-left corners covering the page with `step` stride, always including the far
    edge even if it does not fall on a stride boundary - otherwise a page whose size is
    not a multiple of `step` loses a strip along its bottom and right edges."""
    ys = list(range(0, max(1, height - PATCH_SIZE + 1), step))
    if not ys or ys[-1] != height - PATCH_SIZE:
        ys.append(max(0, height - PATCH_SIZE))
    xs = list(range(0, max(1, width - PATCH_SIZE + 1), step))
    if not xs or xs[-1] != width - PATCH_SIZE:
        xs.append(max(0, width - PATCH_SIZE))
    return [(y, x) for y in sorted(set(ys)) for x in sorted(set(xs))]


@torch.no_grad()
def predict_mask(
    model: CamVidModel, image: np.ndarray, device: str, batch_size: int = 16
) -> np.ndarray:
    """Per-pixel class probabilities for the whole page, tiles averaged in overlap."""
    height, width = image.shape[:2]
    origins = tile_origins(height, width)
    order = model.detector_class_order
    prob_sum = np.zeros((len(order) + 1, height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.float32)

    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start : start + batch_size]
        tiles = np.stack([extract_patch(image, origin, 255) for origin in batch_origins])
        tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float().to(device) / 255.0
        logits = model(tensor)
        if logits.shape[1] != len(order) + 1:
            raise ValueError("checkpoint output channels do not match its detector class order")
        probs = logits.softmax(dim=1).cpu().numpy()

        for (y, x), tile_probs in zip(batch_origins, probs, strict=True):
            y_end, x_end = min(y + PATCH_SIZE, height), min(x + PATCH_SIZE, width)
            prob_sum[:, y:y_end, x:x_end] += tile_probs[:, : y_end - y, : x_end - x]
            coverage[y:y_end, x:x_end] += 1.0

    coverage = np.maximum(coverage, 1.0)
    return prob_sum / coverage[None, :, :]


#: Smallest region worth calling a box. A text box in this corpus is hundreds of pixels;
#: the default used to be 4, which admits a 2x2 blob, so argmax speckle on a 2500x3500 page
#: became thousands of "detections" - 3,749 per page on the released non-lyric-text
#: checkpoint, against ground truth holding one or two.
#:
#: Measured over four pages, sweeping this value with each page's mask computed once:
#:
#:      min_area   predicted   precision
#:             4       6,252        0.1%
#:           200         331        1.2%
#:           800         170        1.8%
#:          2000          94        0.0%   (drops every true box)
#:
#: 200 is where the fragment count falls by 20x while every matched box survives. It does
#: not rescue a bad checkpoint - that sweep was run on one - but it stops a good one from
#: being scored against its own speckle.
MIN_BOX_AREA = 200


def boxes_from_probs(
    probs: np.ndarray,
    min_area: int = MIN_BOX_AREA,
    class_order: tuple[str, ...] = CLASS_ORDER,
) -> list[PredictedBox]:
    """One box per connected foreground region, per class - mirrors
    `detector_masks.rasterize`'s ground-truth shape so the two are directly comparable.
    """
    if probs.shape[0] != len(class_order) + 1:
        raise ValueError("probability channels do not match detector class order")
    class_map = probs.argmax(axis=0).astype(np.uint8)
    confidence_map = probs.max(axis=0)
    boxes = []
    for class_index, label in enumerate(class_order, start=1):
        binary = (class_map == class_index).astype(np.uint8)
        if binary.sum() == 0:
            continue
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for component in range(1, count):  # component 0 is background
            left, top, w, h, area = stats[component]
            if area < min_area:
                continue
            region = binary[top : top + h, left : left + w] > 0
            confidence = float(confidence_map[top : top + h, left : left + w][region].mean())
            boxes.append(
                PredictedBox(label, int(left), int(top), int(left + w), int(top + h), confidence)
            )
    return boxes


def predict_boxes(
    model: CamVidModel, image_path: Path, device: str, batch_size: int = 16
) -> list[PredictedBox]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    probs = predict_mask(model, image, device, batch_size)
    return boxes_from_probs(probs, class_order=model.detector_class_order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--classes", help="Comma-separated order for older weights without a sidecar"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    order = tuple(args.classes.split(",")) if args.classes else None
    model = load_model(args.weights, args.device, order)
    boxes = predict_boxes(model, args.image, args.device)
    by_class: dict[str, int] = {}
    for box in boxes:
        by_class[box.label] = by_class.get(box.label, 0) + 1
    print(f"{len(boxes)} boxes: " + ", ".join(f"{k}={v}" for k, v in sorted(by_class.items())))

    if args.out:
        args.out.write_text(json.dumps([box.__dict__ for box in boxes], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
