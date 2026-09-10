"""
Cut the boxed syllables out of their pages, as a recogniser's training set.

27.45 settled the text pass as detection then recognition. `musescore_boxes` produces what
the detector needs - a page and the typed boxes on it. This produces what the recogniser
needs: one image per syllable and the string it says.

**The split is by score, not by system or by crop.** A Lied's systems share its engraving,
its typesetting and most of its words - *"Es"*, *"und"*, *"die"* recur through a song - so a
crop-level split would put the same syllable, in the same font, at the same size, on both
sides. The result would measure memorisation and report it as recognition. 13.5 took score
level splits for OSSQ for this reason and the same reasoning applies harder here, because
text repeats far more than music does.

Crops keep their native resolution. 27.47 sampled that resolution across the range the
scans span on purpose, and normalising here would throw away the thing that was just paid
for. Height normalisation belongs in the training loop, where it can be part of the
augmentation rather than baked into the corpus.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2

#: Pixels of page kept around a syllable. Recognisers do better with a little air than with
#: a box cut exactly to the ink, and a hyphen sitting just outside the box is worth seeing -
#: it is what distinguishes a word's last syllable from its middle one.
MARGIN = 4

#: A crop smaller than this is not a syllable anyone could read; it means the box was wrong.
MIN_SIDE = 4


@dataclass(frozen=True)
class Crop:
    path: Path
    text: str
    syllabic: str
    verse: str
    score: str


def score_of(system: str) -> str:
    """The score a system belongs to.

    Systems are named `<score>_p<page>-s<system>`, so the score is everything before the
    first underscore. Splitting on the wrong boundary here is the whole leak.
    """
    return system.split("_", 1)[0]


def split_scores(scores: list[str], valid_share: float = 0.15) -> tuple[set[str], set[str]]:
    """Assign whole scores to train or valid, deterministically by name.

    Sorted then dealt, rather than shuffled, so the split is reproducible without carrying a
    seed around and stable as the corpus grows.
    """
    ordered = sorted(set(scores))
    stride = max(2, round(1 / valid_share)) if valid_share > 0 else 0
    valid = {name for index, name in enumerate(ordered) if stride and index % stride == 0}
    return set(ordered) - valid, valid


def crop_syllables(record_path: Path, out_dir: Path) -> list[Crop]:
    """Cut every syllable out of one annotated system."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    page = cv2.imread(str(record_path.parent / record["image"]))
    if page is None:
        return []

    system = record_path.name[: -len(".boxes.json")]
    height, width = page.shape[:2]
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = []
    for index, box in enumerate(record["lyrics"]):
        left = max(0, box["left"] - MARGIN)
        top = max(0, box["top"] - MARGIN)
        right = min(width, box["right"] + MARGIN)
        bottom = min(height, box["bottom"] + MARGIN)
        if right - left < MIN_SIDE or bottom - top < MIN_SIDE:
            continue
        path = out_dir / f"{system}-{index:03d}.png"
        cv2.imwrite(str(path), page[top:bottom, left:right])
        crops.append(
            Crop(path, box["text"], box.get("syllabic", "single"), box.get("verse", "1"),
                 score_of(system))
        )
    return crops


def write_manifest(crops: list[Crop], path: Path) -> None:
    """One JSON object per line: the crop, what it says, and where it came from."""
    with path.open("w", encoding="utf-8") as handle:
        for crop in crops:
            handle.write(
                json.dumps(
                    {
                        "image": str(crop.path),
                        "text": crop.text,
                        "syllabic": crop.syllabic,
                        "verse": crop.verse,
                        "score": crop.score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def describe(train: list[Crop], valid: list[Crop]) -> str:
    alphabet = {character for crop in train + valid for character in crop.text}
    heights = []
    for crop in (train + valid)[:2000]:
        image = cv2.imread(str(crop.path))
        if image is not None:
            heights.append(image.shape[0])

    train_words = {crop.text for crop in train}
    unseen = [crop for crop in valid if crop.text not in train_words]

    lines = [
        f"train {len(train):,} crops from {len({c.score for c in train})} scores",
        f"valid {len(valid):,} crops from {len({c.score for c in valid})} scores",
        f"alphabet {len(alphabet)} characters",
    ]
    if heights:
        lines.append(
            f"crop height median {statistics.median(heights):.0f}px, "
            f"range {min(heights)}-{max(heights)}"
        )
    if valid:
        lines.append(
            f"valid syllables never seen in train: {len(unseen):,} "
            f"({len(unseen) / len(valid):.1%}) - the number a closed vocabulary would miss"
        )
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
    for record in records:
        name = record.name[: -len(".boxes.json")]
        target = "valid" if score_of(name) in valid_scores else "train"
        found = crop_syllables(record, args.out / target)
        (valid if target == "valid" else train).extend(found)

    args.out.mkdir(parents=True, exist_ok=True)
    write_manifest(train, args.out / "train.jsonl")
    write_manifest(valid, args.out / "valid.jsonl")
    print(describe(train, valid))

    scripts = collections.Counter(
        unicodedata.category(character)
        for crop in train + valid
        for character in crop.text
    )
    print("character categories: " + ", ".join(f"{k}={v:,}" for k, v in scripts.most_common(6)))


if __name__ == "__main__":
    main()
