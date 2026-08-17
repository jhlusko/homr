"""
Duplicate the faintest scanned crops in a training index, rather than filtering or
transforming any image.

27.60 found the scan gap concentrated by score - a 70x spread in collapse rate between the
best and the worst of nine documents - and 27.63 ruled out a page-level fix: CLAHE contrast
normalisation damages the crisp scans faster than it helps the faint ones. 27.60's own shape
points at a different lever: change *which examples* the model sees more of, not what each
image looks like.

**This module was first built to key on score identity, using collapse rates measured from
predictions on the validation split - and that design cannot work.** OSSQ's split is by
score (13.5, for the reason every score-level split in this project is: a crop-level split
leaks the same engraving onto both sides). Checked before running a training job on it: the
nine scores with measured collapse rates are all in validation, and there is **zero overlap**
with the training index's score names. Repeating "scores that collapse" in the training
index would match nothing, because the collapsing scores are not in it.

**The fix is to measure the signal directly on training images, with no model and no
validation dependency.** 27.61 found contrast and ink fraction correlate with collapse rate
from the image alone - r=-0.51 and r=-0.56 across nine scores, suggestive rather than
established, but requiring no predictions to compute. Applying that measurement to every
training crop sidesteps the split entirely: it does not care which score a crop belongs to,
only what the crop looks like. It is also finer-grained than the score-level version - 27.60
found 45.8% of a "bad" score's own staves are unaffected, so weighting by score would have
boosted good pages inside bad scores for no reason a per-image weight avoids.

**Oversampling, not filtering.** Dropping the faintest crops would shrink an already-small
scanned corpus and remove exactly the examples a deployed system will meet - a scan that
reads like the worst crops here is not hypothetical, it is what a phone photograph looks
like. Repeating their lines gives the loss more chances to fit them without discarding
anything or touching the paired synthetic set, whose lines are untouched by construction:
this only reads image files under the scanned track's own directories.

This changes an index file, which is data, not a model - no GPU needed to build or verify it,
only to find out whether it helps.
"""

# flake8: noqa: T201

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2


def ink_fraction(image, threshold: int = 128) -> float:
    return float((image < threshold).mean())


def contrast(image) -> float:
    """The 5th-to-95th percentile spread - robust to a handful of outlier pixels."""
    import numpy as np

    return float(np.percentile(image, 95) - np.percentile(image, 5))


@dataclass(frozen=True)
class ImageWeight:
    path: str
    contrast: float
    repeats: int


def measure_contrast(image_paths: list[str]) -> dict[str, float]:
    """Contrast per image path, skipping any that cannot be read rather than raising.

    A handful of missing or corrupt files should not abort weighting the rest - the caller
    treats an unmeasured image the same as an unremarkable one, at weight 1.
    """
    found = {}
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            found[path] = contrast(image)
    return found


def repeat_counts(
    values: dict[str, float], floor_percentile: float = 25.0, max_repeats: int = 4
) -> dict[str, ImageWeight]:
    """How many times each image's line should appear, faintest getting the most.

    Floor is a percentile of the observed distribution, not a fixed contrast value - a
    percentile is stable across corpora with different average scan quality, where a fixed
    threshold would need re-tuning for every new source. Below the floor an image keeps its
    natural weight of one; between the floor and the observed minimum, weight rises linearly
    to the cap. Capped for the reason 27.50 and 27.64 both capped their multipliers: an
    unbounded weight on a handful of the faintest crops risks the model memorising those
    crops rather than learning what makes faint text readable.
    """
    if not values:
        return {}
    ordered = sorted(values.values())
    floor = ordered[min(len(ordered) - 1, int(len(ordered) * floor_percentile / 100))]
    worst = ordered[0]

    weights = {}
    for path, value in values.items():
        if value >= floor or floor <= worst:
            repeats = 1
        else:
            span = max(1e-9, floor - worst)
            repeats = 1 + round((max_repeats - 1) * (floor - value) / span)
        weights[path] = ImageWeight(path, value, min(max_repeats, max(1, repeats)))
    return weights


def reweight_index(index: Path, weights: dict[str, ImageWeight], out: Path) -> tuple[int, int]:
    """Write a new index with each line repeated by its image's weight.

    An image absent from `weights` - unreadable, or outside the measured set - is kept
    once. Guessing a repeat count for an unmeasured image would be acting on a weight that
    was never computed.
    """
    lines = index.read_text(encoding="utf-8").splitlines()
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for line in lines:
            if not line.strip():
                continue
            image_path = line.split(",")[0]
            repeats = weights[image_path].repeats if image_path in weights else 1
            for _ in range(repeats):
                handle.write(line + "\n")
                written += 1
    return len(lines), written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="The training index to reweight.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--floor-percentile", type=float, default=25.0)
    parser.add_argument("--max-repeats", type=int, default=4)
    args = parser.parse_args()

    image_paths = [
        line.split(",")[0] for line in args.index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    values = measure_contrast(image_paths)
    weights = repeat_counts(values, args.floor_percentile, args.max_repeats)

    boosted = [w for w in weights.values() if w.repeats > 1]
    print(f"{len(values):,} images measured, {len(boosted):,} boosted above x1")
    if values:
        print(
            f"  contrast: min {min(values.values()):.0f}  median {statistics.median(values.values()):.0f}"
            f"  max {max(values.values()):.0f}"
        )
    if boosted:
        for weight in sorted(boosted, key=lambda w: w.contrast)[:6]:
            print(f"  {Path(weight.path).name}: contrast {weight.contrast:.0f}  x{weight.repeats}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    before, after = reweight_index(args.index, weights, args.out)
    print(f"\n{before:,} lines -> {after:,} lines ({args.out})")


if __name__ == "__main__":
    main()
