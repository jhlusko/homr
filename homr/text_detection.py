"""Page-level inference for the Stage 3 text detectors, in the shipping runtime.

`text_detector_config` has documented since the detectors were pinned that "no inference
class reads these paths yet; only `download_weights` does". This is that class. Until it
existed, both checkpoints were downloaded on every install and then never executed, so
the fusion design's own summary of the runtime ("already loads both detector checkpoints,
but executes only the selected one and discards its boxes") described something the code
did not do - it loaded neither.

Deliberately a mirror of `training/ocr/detector_inference.py` rather than an import of
it. That module is the reference implementation and depends on torch and
`pytorch_lightning` through `CamVidModel`; the runtime ships neither and reads ONNX. So
the geometry is reimplemented here and pinned to the reference by
`tests/test_text_detection.py`, which asserts tile origins, patch padding and box
recovery agree with `training.ocr` exactly. Predicted boxes and ground-truth boxes stay
comparable by construction, which is the property `detector_inference`'s docstring cares
about and the one a silent divergence here would destroy.

What this module does NOT do, on purpose:

- **No class gating.** A graph can emit a label its loss never trained, and the pinned
  allowlist that rejects those lives in the fusion policy alongside the calibration it is
  versioned with (`fusion_policy.CLASS_ALLOWLIST`). Filtering here as well would put the
  same rule in two places and let them drift apart.
- **No calibration.** `confidence` is the mean maximum pixel probability inside the
  connected component - a raw score, not a probability of a correct box, and not
  comparable across checkpoints or classes. Naming it `raw_confidence` at the boundary is
  the fusion layer's job; naming it anything more confident here would be a lie.
- **No detector selection.** Which detector is correct for a page is a decision with
  evidence behind it, not a default this module should quietly make.
"""

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

from homr.type_definitions import NDArray

#: Class 0 is background; 1..N follow this order. Must equal
#: `training.ocr.detector_masks.CLASS_ORDER` - the exported graph's channel order is
#: this list, so a divergence silently relabels every box rather than failing.
CLASS_ORDER = (
    "Dynamic",
    "Fingering",
    "Expression",
    "Tempo",
    "MeasureNumber",
    "StaffText",
    "Lyrics",
)

BACKGROUND = 0
CLASS_INDEX = {name: index + 1 for index, name in enumerate(CLASS_ORDER)}
NUM_CLASSES = len(CLASS_ORDER) + 1

PATCH_SIZE = 320
STEP = PATCH_SIZE // 2

#: Components smaller than this are dropped. Matches the reference implementation.
MIN_AREA = 4


@dataclass(frozen=True)
class DetectedBox:
    """One connected component, in source-page pixels, right/bottom exclusive."""

    label: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float


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


def extract_patch(array: NDArray, origin: tuple[int, int], pad_value: int) -> NDArray:
    """A `PATCH_SIZE` square from `array` at `origin`, padded past the page edge.

    Pads rather than shrinks, so every patch the model sees is the same shape regardless
    of where on the page it fell.
    """
    y, x = origin
    height, width = array.shape[:2]
    channels = array.shape[2:]
    patch = np.full((PATCH_SIZE, PATCH_SIZE, *channels), pad_value, dtype=array.dtype)
    y_end, x_end = min(y + PATCH_SIZE, height), min(x + PATCH_SIZE, width)
    patch[: y_end - y, : x_end - x] = array[y:y_end, x:x_end]
    return patch


def _softmax(logits: NDArray, axis: int) -> NDArray:
    """Softmax over `axis`, shifted by the max for numerical stability.

    The exported graph ends at the model's raw output, matching `convert_segnet`'s export
    shape, so the runtime applies the same `logits.softmax(dim=1)` the reference does
    after inference rather than inside the graph.
    """
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=axis, keepdims=True)


class TextDetector:
    """One pinned detector checkpoint, executed over whole pages.

    A session is not thread-safe to create concurrently, and page inference allocates a
    `(NUM_CLASSES, height, width)` float32 accumulator - about 190 MB for a 3000x2000
    page - so instances are cached per path rather than per call.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"text detector weights not found: {model_path}. These are optional "
                "downloads; see homr.text_detector_config."
            )
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_mask(self, image: NDArray, batch_size: int = 16) -> NDArray:
        """Per-pixel class probabilities for the whole page, tiles averaged in overlap.

        Averaging rather than letting one tile win at a seam: with 50% overlap every
        interior pixel is predicted twice, and picking one arbitrarily puts a visible
        discontinuity exactly where a box boundary is most likely to be judged.
        """
        height, width = image.shape[:2]
        origins = tile_origins(height, width)
        prob_sum = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=np.float32)

        for start in range(0, len(origins), batch_size):
            batch_origins = origins[start : start + batch_size]
            tiles = np.stack([extract_patch(image, origin, 255) for origin in batch_origins])
            # (batch, y, x, channel) -> (batch, channel, y, x), scaled to [0, 1]. The
            # reference reads pages with cv2.imread and does not convert colour, so the
            # channel order the model was trained on is BGR; converting here would be a
            # silent input distribution shift.
            tensor = np.transpose(tiles, (0, 3, 1, 2)).astype(np.float32) / 255.0
            logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
            probs = _softmax(np.asarray(logits, dtype=np.float32), axis=1)

            for (y, x), tile_probs in zip(batch_origins, probs, strict=True):
                y_end, x_end = min(y + PATCH_SIZE, height), min(x + PATCH_SIZE, width)
                prob_sum[:, y:y_end, x:x_end] += tile_probs[:, : y_end - y, : x_end - x]
                coverage[y:y_end, x:x_end] += 1.0

        coverage = np.maximum(coverage, 1.0)
        return prob_sum / coverage[None, :, :]

    def detect(self, image: NDArray, batch_size: int = 16) -> list[DetectedBox]:
        return boxes_from_probs(self.predict_mask(image, batch_size))


def boxes_from_probs(probs: NDArray, min_area: int = MIN_AREA) -> list[DetectedBox]:
    """One box per connected foreground region, per class.

    Mirrors the ground truth's own `connectedComponentsWithStats` shape, so a predicted
    box and a ground-truth box are comparable by construction rather than by convention.
    """
    class_map = probs.argmax(axis=0).astype(np.uint8)
    confidence_map = probs.max(axis=0)
    boxes = []
    for label, class_index in CLASS_INDEX.items():
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
                DetectedBox(label, int(left), int(top), int(left + w), int(top + h), confidence)
            )
    return boxes


_detector_cache: dict[str, TextDetector] = {}
_cache_lock = threading.Lock()


def get_detector(model_path: str) -> TextDetector:
    """Cached detector for `model_path`. Both detectors coexist by design - fusion runs
    them over the identical page, so neither may evict the other."""
    with _cache_lock:
        if model_path not in _detector_cache:
            _detector_cache[model_path] = TextDetector(model_path)
        return _detector_cache[model_path]


def detect_text(image: NDArray, model_path: str, batch_size: int = 16) -> list[DetectedBox]:
    return get_detector(model_path).detect(image, batch_size)
