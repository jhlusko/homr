"""
Cut the boxed Dynamic marks out of their pages, as a classifier's training set.

27.94 decided dynamics need a small closed-set classifier, not the CTC recogniser -
`p`/`f`/`mf`/... is a discrete choice among about a dozen marks, not open-vocabulary text.
27.95 built the detector half (find the box); this is the data half of reading it: one
crop per Dynamic box, and the mark it is.

**The label comes from the source MusicXML, joined by position - and the order is a real
risk, checked rather than assumed.** `musescore_boxes.boxes_of_class` orders boxes in
*reading order* (line by line, top to bottom, then left to right); `source_dynamics`
orders marks in raw XML document order via `root.iter("dynamics")`. These are not the same
thing in general - it is exactly the shape of bug that has hit this project three times
before (27.11, 27.15, 27.17), and `source_syllables`' own docstring records a real instance
of document order and reading order actually diverging (multi-verse lyrics). Dynamics do
not have that specific problem (no verse interleaving), but nothing here proves reading
order and document order coincide for a Dynamic mark spanning an unusual layout, so the
join refuses outright on a count mismatch - the same discipline `musescore_boxes.pair`
already applies to lyrics - rather than trusting position silently.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2

from training.omr_datasets.musescore_boxes import source_dynamics

#: Pixels of page kept around a mark. Same reasoning as lyric_crops.py's MARGIN: a
#: classifier reads better with a little air than a box cut exactly to the ink.
MARGIN = 4

#: A crop smaller than this is not a mark anyone could classify; it means the box was wrong.
MIN_SIDE = 4


@dataclass(frozen=True)
class Crop:
    path: Path
    label: str
    score: str


def score_of(system: str) -> str:
    """The score a system belongs to - see lyric_crops.py's `score_of` for why this is
    the split boundary, not the system or the crop."""
    return system.split("_", 1)[0]


def split_scores(scores: list[str], valid_share: float = 0.15) -> tuple[set[str], set[str]]:
    ordered = sorted(set(scores))
    stride = max(2, round(1 / valid_share)) if valid_share > 0 else 0
    valid = {name for index, name in enumerate(ordered) if stride and index % stride == 0}
    return set(ordered) - valid, valid


def crop_dynamics(record_path: Path, out_dir: Path) -> tuple[list[Crop], str]:
    """Cut every Dynamic mark out of one annotated system.

    Returns (crops, refusal-reason) - refusal is empty on success, matching
    `musescore_boxes.Unrenderable`'s reason-string convention rather than raising, since a
    mismatch here is expected to happen sometimes and the caller counts it rather than
    stopping the whole corpus build.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    boxes = record.get("text_boxes", {}).get("Dynamic", [])
    if not boxes:
        return [], ""

    system = record_path.name[: -len(".boxes.json")]
    score_path = record_path.parent / f"{system}.render.musicxml"
    if not score_path.is_file():
        return [], "no .render.musicxml"

    try:
        labels = source_dynamics(score_path)
    except ET.ParseError:
        return [], "unparseable musicxml"

    if len(boxes) != len(labels):
        return [], f"{len(boxes)} rendered boxes against {len(labels)} source dynamics"

    page = cv2.imread(str(record_path.parent / record["image"]))
    if page is None:
        return [], "cannot read image"
    height, width = page.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    for index, (box, label) in enumerate(zip(boxes, labels, strict=True)):
        left = max(0, box["left"] - MARGIN)
        top = max(0, box["top"] - MARGIN)
        right = min(width, box["right"] + MARGIN)
        bottom = min(height, box["bottom"] + MARGIN)
        if right - left < MIN_SIDE or bottom - top < MIN_SIDE:
            continue
        path = out_dir / f"{system}-{index:03d}.png"
        cv2.imwrite(str(path), page[top:bottom, left:right])
        crops.append(Crop(path, label, score_of(system)))
    return crops, ""


def write_manifest(crops: list[Crop], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for crop in crops:
            handle.write(
                json.dumps(
                    {"image": str(crop.path), "label": crop.label, "score": crop.score},
                    ensure_ascii=False,
                )
                + "\n"
            )


def describe(train: list[Crop], valid: list[Crop], refused: collections.Counter) -> str:
    labels = collections.Counter(c.label for c in train + valid)
    heights = []
    for crop in (train + valid)[:2000]:
        image = cv2.imread(str(crop.path))
        if image is not None:
            heights.append(image.shape[0])

    lines = [
        f"train {len(train):,} crops from {len({c.score for c in train})} scores",
        f"valid {len(valid):,} crops from {len({c.score for c in valid})} scores",
        f"{len(labels)} distinct labels: " + ", ".join(f"{k}={v}" for k, v in labels.most_common()),
    ]
    if heights:
        lines.append(
            f"crop height median {statistics.median(heights):.0f}px, "
            f"range {min(heights)}-{max(heights)}"
        )
    if refused:
        lines.append(f"{sum(refused.values())} systems refused: {dict(refused.most_common(6))}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--boxes", type=Path, required=True, help="A musescore_boxes out dir.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--valid-share", type=float, default=0.15)
    args = parser.parse_args()

    records = sorted(args.boxes.glob("*/*.boxes.json"))
    if not records:
        raise SystemExit(f"No .boxes.json under {args.boxes}")

    systems = [path.name[: -len(".boxes.json")] for path in records]
    train_scores, valid_scores = split_scores([score_of(s) for s in systems], args.valid_share)

    train: list[Crop] = []
    valid: list[Crop] = []
    refused: collections.Counter = collections.Counter()
    for record in records:
        name = record.name[: -len(".boxes.json")]
        target = "valid" if score_of(name) in valid_scores else "train"
        found, reason = crop_dynamics(record, args.out / target)
        if reason:
            bucket = "count mismatch" if "boxes against" in reason else reason
            refused[bucket] += 1
        (valid if target == "valid" else train).extend(found)

    args.out.mkdir(parents=True, exist_ok=True)
    write_manifest(train, args.out / "train.jsonl")
    write_manifest(valid, args.out / "valid.jsonl")
    print(describe(train, valid, refused))


if __name__ == "__main__":
    main()
