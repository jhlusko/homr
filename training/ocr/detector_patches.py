"""
Sample 320x320 training patches biased toward the boxes, not tiled uniformly across the page.

27.69 measured text coverage at 0.38% of a page. homr's own segmentation training
(`training/segmentation/dense_dataset_definitions.py`) tiles a plain non-overlapping grid of
320x320 patches across the whole page with no bias toward content, which is right for its
own targets - noteheads and staff lines are dense enough that most tiles contain some. Text
is not: a typical page here holds around 13 boxes (27.68 - 37,356 boxes over 2,847 images),
and a 4500x3200 page tiles into on the order of 140 non-overlapping 320x320 windows, so an
unbiased grid would put text in a handful of tiles and nothing in the rest. Copying homr's
sampler here would train a detector almost entirely on the answer "nothing here" - which
28.62's global-focal mistake already showed happens by accident when a correction is applied
somewhere it was not measured to be needed; this is the same trap approached from the data
side, avoided by measuring first rather than reusing a pattern built for different data.

**Sampling, not tiling.** Each drawn patch is centred either on a randomly chosen box (with
jitter, so a box is not always dead-centre) or on a uniformly random page location, at a
fixed ratio - the positive samples give the loss something to learn from every step, the
negative samples keep the model from predicting text everywhere out of habit.

Edge padding follows the convention already set for this corpus: white for the image (paper,
not ink - 27.51's recogniser found the alternative teaches the model that every crop ends in
a black bar), background class for the mask.
"""

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

#: Matches homr's own segmentation patch size (training/segmentation/train.py), so a
#: detector trained on this data could share inference-time tiling logic with the existing
#: segmentation model if that turns out to be worth doing later.
PATCH_SIZE = 320

#: Share of drawn patches centred on a box rather than a random location.
POSITIVE_RATIO = 0.7

#: Fraction of the patch size a positive sample's centre may drift from the box centre, so
#: the box is not always dead-centre - a detector trained on perfectly centred boxes only
#: would not learn to find one near a patch edge.
JITTER = 0.3

PAD_IMAGE_VALUE = 255
PAD_MASK_VALUE = 0


@dataclass(frozen=True)
class Sample:
    image: str
    mask: str


def read_index(path: Path) -> list[Sample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, mask = line.split(",")
        samples.append(Sample(image, mask))
    return samples


def box_centres(mask: np.ndarray) -> list[tuple[int, int]]:
    """Centre of every connected foreground region in a class mask, in (y, x)."""
    binary = (mask != PAD_MASK_VALUE).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # Label 0 is always the background component; connectedComponentsWithStats guarantees it.
    return [(int(round(cy)), int(round(cx))) for cx, cy in centroids[1:count]]


def patch_origin(
    center: tuple[int, int], shape: tuple[int, int], jitter: float, rng: random.Random
) -> tuple[int, int]:
    """Top-left corner of a patch around `center`, jittered and clamped to the page."""
    height, width = shape
    max_shift = int(PATCH_SIZE * jitter)
    y = center[0] - PATCH_SIZE // 2 + rng.randint(-max_shift, max_shift)
    x = center[1] - PATCH_SIZE // 2 + rng.randint(-max_shift, max_shift)
    return max(0, min(y, max(0, height - PATCH_SIZE))), max(0, min(x, max(0, width - PATCH_SIZE)))


def extract_patch(
    array: np.ndarray, origin: tuple[int, int], pad_value: int
) -> np.ndarray:
    """A `PATCH_SIZE` square from `array` at `origin`, padded past the page edge.

    Mirrors `SegmentationBaseDataset._get_patch`'s edge behaviour - pad rather than shrink,
    so every patch the model sees is the same shape regardless of where on the page it fell.
    """
    y, x = origin
    height, width = array.shape[:2]
    channels = array.shape[2:]
    patch = np.full((PATCH_SIZE, PATCH_SIZE, *channels), pad_value, dtype=array.dtype)
    y_end, x_end = min(y + PATCH_SIZE, height), min(x + PATCH_SIZE, width)
    patch[: y_end - y, : x_end - x] = array[y:y_end, x:x_end]
    return patch


class DetectorPatches(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        patches_per_image: int = 8,
        positive_ratio: float = POSITIVE_RATIO,
        seed: int = 0,
    ) -> None:
        self.samples = samples
        self.patches_per_image = patches_per_image
        self.positive_ratio = positive_ratio
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[index // self.patches_per_image]
        image = cv2.imread(sample.image)
        mask = cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"cannot read {sample.image} or {sample.mask}")

        centres = box_centres(mask)
        positive = centres and self.rng.random() < self.positive_ratio
        if positive:
            center = self.rng.choice(centres)
        else:
            center = (
                self.rng.randint(0, mask.shape[0] - 1), self.rng.randint(0, mask.shape[1] - 1)
            )
        origin = patch_origin(center, mask.shape[:2], JITTER, self.rng)

        return (
            extract_patch(image, origin, PAD_IMAGE_VALUE),
            extract_patch(mask, origin, PAD_MASK_VALUE),
        )
