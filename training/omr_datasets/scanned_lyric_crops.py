"""Syllable crops cut from *scans*, for a recogniser that has only ever seen renders.

`lyric_crops` builds the recogniser's corpus by rendering OpenScore Lieder through
MuseScore. 27.38 called that the easy case and predicted the cost -- "a lyric recogniser
trained on rendered text would overstate by more" than the 25 points the notation heads
lost crossing to scanned -- and `eval_recognizer_on_scans.py` measured it: exact match
falls from 88.7% to 13.0%, CER rises from 11.6% to 61.0%.

This builds the same manifest from the same songs *photographed* instead. `ocr_first`'s
matches carry a box, the syllable it says, and the page it sits on, so the crops are real
scanned type with labels that were confirmed against the score's own MusicXML.

Two rules inherited deliberately from `lyric_crops`, because a corpus that breaks them
measures memorisation and reports it as recognition:

  * **Split by score, not by crop.** A Lied's systems share its engraving, its typesetting
    and most of its words, so a crop-level split puts the same syllable in the same font
    on both sides of it.
  * **Only exact matches.** A partial OCR match is not a label. Training on one teaches
    the model to read something the page does not say.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2

from training.omr_datasets.lyric_crops import split_scores

#: Padding around the OCR box, in pixels.
#:
#: The boxes come from RapidOCR's own text detection, which hugs the ink. `lyric_crops`
#: cuts from MuseScore's glyph boxes, which include the font's side bearings, so a crop
#: with no margin here would be systematically tighter than what the recogniser was built
#: to read -- a difference in framing dressed up as a difference in domain.
MARGIN = 3


@dataclass(frozen=True)
class ScannedCrop:
    path: Path
    text: str
    score: str
    verse: str


def crops_for_score(
    matches_path: Path, pages_root: Path, out_dir: Path, margin: int = MARGIN
) -> tuple[list[ScannedCrop], collections.Counter]:
    data = json.loads(matches_path.read_text(encoding="utf-8"))
    score = data["score_id"]
    skipped: collections.Counter = collections.Counter()
    crops: list[ScannedCrop] = []
    page_cache: dict[str, object] = {}

    for index, match in enumerate(data.get("matches", [])):
        if match.get("kind") != "lyric":
            skipped["not-a-lyric"] += 1
            continue
        # An inexact match is not a label; see the module docstring.
        if float(match.get("matched_fraction", 0)) < 1.0:
            skipped["partial-match"] += 1
            continue
        page_name = match.get("page_image")
        page_path = pages_root / score / str(page_name)
        if not page_path.exists():
            skipped["page-missing"] += 1
            continue
        if page_name not in page_cache:
            page_cache.clear()
            page_cache[page_name] = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
        page = page_cache[page_name]
        if page is None:
            skipped["page-unreadable"] += 1
            continue

        box = match["box"]
        height, width = page.shape[:2]
        left = max(0, int(box["left"]) - margin)
        top = max(0, int(box["top"]) - margin)
        right = min(width, int(box["left"]) + int(box["width"]) + margin)
        bottom = min(height, int(box["top"]) + int(box["height"]) + margin)
        if right - left < 4 or bottom - top < 8:
            skipped["degenerate-box"] += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{score}_{index:05d}.png"
        cv2.imwrite(str(target), page[top:bottom, left:right])
        crops.append(
            ScannedCrop(
                path=target,
                # Normalised so the same syllable is one label however the source encoded
                # its accents; an alphabet built from mixed forms carries both.
                text=unicodedata.normalize("NFC", match["text"]),
                score=score,
                verse=str(match.get("verse", "")),
            )
        )
    return crops, skipped


def write_manifest(crops: list[ScannedCrop], path: Path) -> None:
    """The shape `recognizer_data.read_manifest` reads, so both corpora load the same."""
    with path.open("w", encoding="utf-8") as handle:
        for crop in crops:
            handle.write(
                json.dumps(
                    {
                        "image": str(crop.path),
                        "text": crop.text,
                        "verse": crop.verse,
                        "score": crop.score,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def describe(train: list[ScannedCrop], valid: list[ScannedCrop]) -> str:
    alphabet = {c for crop in train + valid for c in crop.text}
    train_vocab = {crop.text for crop in train}
    unseen = [crop for crop in valid if crop.text not in train_vocab]
    lengths = sorted(len(crop.text) for crop in train + valid)
    lines = [
        f"{len(train):,} train crops, {len(valid):,} valid, "
        f"{len({c.score for c in train})}/{len({c.score for c in valid})} scores",
        f"alphabet {len(alphabet)} characters",
        # The number that decides whether a score is recognition or recall; see
        # `train_recognizer`'s own docstring.
        f"{len(unseen):,} of {len(valid):,} valid syllables are unseen in training "
        f"({len(unseen)/max(1,len(valid)):.1%})",
        f"syllable length median {lengths[len(lengths)//2] if lengths else 0}, "
        f"max {lengths[-1] if lengths else 0}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--matches", type=Path, required=True, help="A lieder_vocal_text directory of <score>.json."
    )
    parser.add_argument("--pages", type=Path, help="Page images root; defaults to <matches>/pages.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--valid-share", type=float, default=0.15)
    args = parser.parse_args()

    pages_root = args.pages or (args.matches / "pages")
    records = sorted(args.matches.glob("*.json"))
    if not records:
        raise SystemExit(f"No <score>.json under {args.matches}")

    scores = [json.loads(p.read_text(encoding="utf-8"))["score_id"] for p in records]
    train_scores, valid_scores = split_scores(scores, args.valid_share)

    train: list[ScannedCrop] = []
    valid: list[ScannedCrop] = []
    skipped: collections.Counter = collections.Counter()
    for record in records:
        score = json.loads(record.read_text(encoding="utf-8"))["score_id"]
        target = "valid" if score in valid_scores else "train"
        found, missed = crops_for_score(record, pages_root, args.out / target)
        skipped.update(missed)
        (valid if target == "valid" else train).extend(found)

    args.out.mkdir(parents=True, exist_ok=True)
    write_manifest(train, args.out / "train.jsonl")
    write_manifest(valid, args.out / "valid.jsonl")
    print(describe(train, valid))
    # Counted rather than silent: a corpus that starts refusing everything should say so.
    print("skipped: " + ", ".join(f"{k}={v:,}" for k, v in skipped.most_common()))


if __name__ == "__main__":
    main()
