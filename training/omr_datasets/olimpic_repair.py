"""
Extend OLiMPiC's system boxes upward, to take in the voice staff and its lyrics.

27.39 measured what the published annotations actually bound: a median 41% of the distance
between consecutive systems, quartiles 38% to 43%. A box covering a whole system would
leave only an inter-system margin, around 80-90%. At 41% these bound the piano grand staff,
and the voice staff with its lyrics sits in the gap above - which is why OLiMPiC's scanned
images contain no lyrics at all.

The engraving order in a Lieder system is voice, then its lyrics, then the piano grand
staff. So the region above a piano box belongs to *that* system, not the one before it, and
recovering it is an upward extension rather than a re-annotation.

The extension stops short of the previous system's lower edge. Stems, ledger lines and
slurs from the piano above routinely overhang its box, and pulling them into this crop
would put ink from one system into another's image - which is the same error as the
crop-to-part misalignment of 27.11, arriving through geometry instead of indexing. A fixed
margin turned out to guess that overhang low, so given the page image the top is trimmed
down to the first clear band instead of assumed.
"""

# flake8: noqa: T201

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

#: Pixels left between an extended box and the system above it. The gap between systems
#: holds the voice staff and lyrics plus margin, so a little is given back to avoid
#: catching the piano's overhanging stems and slurs from the system above.
SAFETY_MARGIN = 24

#: For the first system on a page there is nothing above to stop at, so the extension uses
#: the page's own typical gap instead. Falls back to the box height when a page has only
#: one system and no gap can be measured.
FALLBACK_GAP_RATIO = 1.4


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def _typical_gap(boxes: list[Box]) -> float:
    """The usual vertical space between one system's box and the next one's top."""
    gaps = [second.top - first.bottom for first, second in zip(boxes, boxes[1:])]
    positive = [gap for gap in gaps if gap > 0]
    return statistics.median(positive) if positive else 0.0


def _ceilings(ordered: list[Box]) -> list[int]:
    """How far up each box may reach: just under the system above, or the page's own gap."""
    fallback = _typical_gap(ordered)
    limits = []
    for index, box in enumerate(ordered):
        if index == 0:
            reach = fallback if fallback > 0 else box.height * FALLBACK_GAP_RATIO
            limits.append(int(round(box.top - reach)))
        else:
            limits.append(ordered[index - 1].bottom + SAFETY_MARGIN)
    return [max(0, min(limit, box.top)) for limit, box in zip(limits, ordered)]


def extend_upward(boxes: list[Box], page: np.ndarray | None = None) -> list[Box]:
    """Grow each box upward into the space its voice staff and lyrics occupy.

    Without a page image this is pure geometry: each box stops `SAFETY_MARGIN` below the
    previous system's lower edge, and the first stops at the page's typical gap above
    itself, having no predecessor to measure against.

    Given the page, the top is then pulled down to the first clear band, which is where the
    system above actually stops rather than where its box says it does. Inspection of the
    geometric version found a sliver of cut-off staff at the top of some crops, and this is
    what removes it.
    """
    ordered = sorted(boxes, key=lambda box: box.top)
    extended: list[Box] = []

    for box, ceiling in zip(ordered, _ceilings(ordered)):
        grown = Box(box.left, ceiling, box.width, box.bottom - ceiling)
        extended.append(grown if page is None else trim_to_gutter(page, box, ceiling))

    return extended


#: A row is blank when fewer than this share of its pixels are ink.
BLANK_ROW_INK = 0.01

#: How many blank rows in a row make a gutter rather than a gap inside a slur or a stem.
GUTTER_ROWS = 8


def trim_to_gutter(page: np.ndarray, box: Box, ceiling: int) -> Box:
    """Pull a box's top down to the first clear band below `ceiling`.

    A fixed margin is a guess about how far the system above overhangs its own box, and
    inspection found it guessing low: stems and slurs from the piano above survived into
    the extended crop as a sliver of cut-off staff at the top edge. The page itself knows
    where the overhang ends - it is where the ink stops - so this reads that instead.

    Falls back to the box it was given when no clear band exists, which happens when the
    voice staff sits directly under the overhang. A crop with a sliver is worth more than
    no crop at all, and 27.39's whole point is that these images have been unusable.
    """
    strip = page[max(0, ceiling) : box.top, box.left : box.left + box.width]
    if strip.size == 0:
        return box

    grey = strip if strip.ndim == 2 else strip.mean(axis=2)
    blank = (grey < 128).mean(axis=1) < BLANK_ROW_INK

    run = 0
    for offset, is_blank in enumerate(blank):
        run = run + 1 if is_blank else 0
        if run >= GUTTER_ROWS:
            top = max(0, ceiling) + offset - run + 1
            return Box(box.left, top, box.width, box.bottom - top)
    return box


def coverage(boxes: list[Box]) -> float:
    """Share of the distance between systems that the boxes occupy.

    The measure 27.39 used, so a repair can be checked in the same terms it was diagnosed
    in rather than by eye.
    """
    ordered = sorted(boxes, key=lambda box: box.top)
    ratios = [
        first.height / (second.top - first.top)
        for first, second in zip(ordered, ordered[1:])
        if second.top > first.top
    ]
    return statistics.median(ratios) if ratios else 0.0


def load_pages(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def repair_document(
    document: dict, pngs: Path | None = None
) -> tuple[dict, list[float], list[float]]:
    """Rewrite one document's boxes; returns it with the before and after coverages.

    Given `pngs`, each page is read so the top can be trimmed to its gutter. Without it the
    repair is geometric, which still recovers the voice staff but leaves a sliver of the
    system above on some pages.
    """
    before, after = [], []
    for page in (document.get("pages") or {}).values():
        systems = [system for system in (page.get("systems") or []) if "boundingBox" in system]
        boxes = [
            Box(**{key: int(value) for key, value in system["boundingBox"].items()})
            for system in systems
        ]
        if len(boxes) < 2:
            continue
        image = None
        if pngs is not None:
            image = cv2.imread(str(pngs.joinpath(*str(page.get("image", "")).split("/"))))
        before.append(coverage(boxes))
        grown = extend_upward(boxes, image)
        after.append(coverage(grown))
        for system, box in zip(sorted(systems, key=lambda s: s["boundingBox"]["top"]), grown):
            system["boundingBox"] = box.to_dict()
    return document, before, after


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--systems", type=Path, required=True, help="An imslp_systems dir.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write repaired yaml.")
    parser.add_argument(
        "--pngs", type=Path, help="An imslp_pngs dir. Given it, tops are trimmed to the gutter."
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    before_all, after_all = [], []
    for path in sorted(args.systems.glob("*.yaml")):
        document, before, after = repair_document(load_pages(path), args.pngs)
        before_all.extend(before)
        after_all.extend(after)
        (args.out / path.name).write_text(yaml.safe_dump(document), encoding="utf-8")

    if before_all:
        print(f"{len(before_all):,} pages repaired")
        print(f"  coverage before {statistics.median(before_all):.0%}")
        print(f"  coverage after  {statistics.median(after_all):.0%}")
        print("  a whole system is around 80-90%; 41% was the piano alone (27.39)")


if __name__ == "__main__":
    main()
