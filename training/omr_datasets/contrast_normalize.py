"""
Does normalising contrast make faint scans look like crisp ones - and does it cost anything?

27.61 found the collapsed staves are faint rather than misaligned, with contrast correlating
with collapse rate at -0.51 across nine scores - suggestive, not established, since a
correlation on nine points carries a p around 0.16. It licensed testing contrast
normalisation without licensing the belief that it would work.

This is the cheap half of that test, and it runs on CPU: does a normalisation transform
actually close the measured gap between the faint scores and the crisp one, or does it do
nothing? Training a model to find out costs GPU hours this project has been careful with
today; measuring the transform's effect on the numbers 27.61 already computed costs a script.

**The transform is CLAHE**, not a global histogram stretch. A global stretch is defeated by
the exact failure mode of a photographed page: uneven illumination means the "faint" and
"crisp" regions can sit on the same page, so one global gain cannot lift one without also
blowing out the other. CLAHE operates on local tiles and is the standard answer to uneven
scan illumination for this reason.

**The risk this file exists to catch**: a transform aggressive enough to fix the worst scans
might visibly damage the best ones - manufacturing noise in already-clean staff lines, or
sharpening scan artifacts into false noteheads. So every score is measured, not only the
faint ones, and the report says whether the crisp scores are left alone.
"""

# flake8: noqa: T201

import argparse
import glob
import statistics
from pathlib import Path

import cv2
import numpy as np


def ink_fraction(image: np.ndarray, threshold: int = 128) -> float:
    return float((image < threshold).mean())


def contrast(image: np.ndarray) -> float:
    """The 5th-to-95th percentile spread - robust to a handful of pure-black or pure-white
    outlier pixels in a way min/max is not."""
    return float(np.percentile(image, 95) - np.percentile(image, 5))


def normalize(image: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE: local contrast equalisation, tile by tile.

    `clip_limit` caps how far any tile's histogram may be stretched, which is what keeps a
    tile that is already high-contrast - a crisp score's staff lines - from being pushed
    into noise. Local rather than global because uneven scan illumination can put a faint
    region and a crisp one on the same page; one global gain cannot serve both.
    """
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return engine.apply(image)


def measure_score(
    paths: list[Path], clip_limit: float = 2.0
) -> tuple[float, float, float, float]:
    """(ink before, ink after, contrast before, contrast after), averaged over the score."""
    before_ink, after_ink, before_con, after_con = [], [], [], []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        normalized = normalize(image, clip_limit=clip_limit)
        before_ink.append(ink_fraction(image))
        after_ink.append(ink_fraction(normalized))
        before_con.append(contrast(image))
        after_con.append(contrast(normalized))
    if not before_ink:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        statistics.mean(before_ink), statistics.mean(after_ink),
        statistics.mean(before_con), statistics.mean(after_con),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--images", type=Path, required=True, help="Dir of scanned crops.")
    parser.add_argument("--sample", type=int, default=40, help="Crops per score to measure.")
    parser.add_argument("--clip-limit", type=float, default=2.0)
    args = parser.parse_args()

    by_score: dict[str, list[Path]] = {}
    for path in sorted(Path(p) for p in glob.glob(str(args.images / "*.png"))):
        by_score.setdefault(path.stem.split("_")[0], []).append(path)

    print(f"{'score':<12} {'ink before':>10} {'ink after':>9}   {'contrast before':>16} {'contrast after':>15}")
    rows = []
    for score, paths in sorted(by_score.items()):
        before_ink, after_ink, before_con, after_con = measure_score(
            paths[: args.sample], args.clip_limit
        )
        rows.append((score, before_ink, after_ink, before_con, after_con))
        print(
            f"{score:<12} {before_ink:>10.3f} {after_ink:>9.3f}   "
            f"{before_con:>16.0f} {after_con:>15.0f}"
        )

    spread_before = max(r[3] for r in rows) - min(r[3] for r in rows)
    spread_after = max(r[4] for r in rows) - min(r[4] for r in rows)
    print(f"\nspread in mean contrast across scores: {spread_before:.0f} before, {spread_after:.0f} after")
    print("  a transform that closes the gap between scores narrows this; one that damages")
    print("  the crisp scores to reach the faint ones would widen it instead")


if __name__ == "__main__":
    main()
