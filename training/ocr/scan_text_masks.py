"""Detector masks from the OCR-first real-scan text ground truth.

`detector_masks.py` rasterises MuseScore box annotations, where every class is
labelled and anything unboxed is genuinely background. Neither holds for
`ocr_first_text_ground_truth.py`'s output, and writing its boxes through that
assumption would teach the detector two false things at once:

1. **Only 2 of 8 classes are present.** Our data has Lyrics and Dynamic. Marking
   everything else background says Tempo, StaffText, Expression, Fingering and
   MeasureNumber do not occur on real scans - and those are precisely the classes
   whose precision has already collapsed (§1).
2. **Even the Lyrics labels are incomplete.** A match exists only where OCR read the
   text *and* it matched the piece's own MusicXML. Real lyrics the OCR missed would
   become background, training the detector to miss lyrics - the exact failure the
   data was gathered to fix.

So the default here writes `IGNORE` outside matched boxes rather than background:
those pixels contribute to no loss term, and the negatives come instead from
replaying the fully-labelled synthetic masks. `smp.losses.DiceLoss` accepts
`ignore_index` directly, so this needs no change to the loss itself.

`--background-outside` reproduces the naive behaviour on purpose, so the choice can
be *measured* rather than argued - the project's own standing rule (see
`train_detector.py`'s docstring: "measure the unweighted baseline before reaching
for class weighting or focal loss") applied to masking.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.ocr.detector_masks import BACKGROUND, CLASS_INDEX

#: uint8 value meaning "no supervision here". Outside 0..len(CLASS_ORDER), and the
#: value handed to DiceLoss as ignore_index.
IGNORE = 255

#: OCR-first `kind` -> detector class name. The last three arrive only from the OSSQ
#: instrumental corpus, whose MuseScore sources carry tempo, staff text and expression
#: markings that the Lieder MusicXML path never produced - see
#: `mscx_text_ground_truth.py`.
KIND_TO_CLASS = {
    "lyric": "Lyrics",
    "dynamic": "Dynamic",
    "tempo": "Tempo",
    "stafftext": "StaffText",
    "expression": "Expression",
}

#: Grayscale level at or above which a pixel is treated as blank paper rather than ink.
#: Deliberately well below pure white, since these are photographed and scanned pages
#: whose paper is rarely 255 and often uneven.
BLANK_THRESHOLD = 200


def mask_for_page(
    width: int,
    height: int,
    matches: list[dict],
    background_outside: bool = False,
    image: np.ndarray | None = None,
    blank_threshold: int = BLANK_THRESHOLD,
) -> np.ndarray:
    """One page's mask: matched boxes carry their class, everything else is
    `IGNORE` (or `BACKGROUND` when `background_outside`, for the ablation).

    Passing `image` selects a third policy, between those two. Marking a whole scan
    page `IGNORE` outside the matched boxes is safe but expensive: measured, it leaves
    only ~2% of pixels supervised, so scan pages teach the detector what text looks
    like while teaching it almost nothing about where text is *absent* - and mixing
    that into training halved page-level precision (§7, E3 box eval) even though it
    improved patch IoU on real scans.

    The recoverable part of that negative signal rests on a fact that needs no OCR to
    be certain of: **blank paper is never text**. A pixel with no ink cannot be a
    lyric the OCR missed, so outside the matched boxes it can be labelled background
    honestly. Inked pixels outside a box stay `IGNORE`, because ink there is genuinely
    ambiguous - it may be notation, or it may be exactly the missed lyric this whole
    masking scheme exists to avoid mislabelling.
    """
    fill = BACKGROUND if background_outside else IGNORE
    mask = np.full((height, width), fill, dtype=np.uint8)
    if image is not None and not background_outside:
        mask[image >= blank_threshold] = BACKGROUND
    for match in matches:
        class_name = KIND_TO_CLASS.get(match.get("kind", ""))
        if class_name is None or class_name not in CLASS_INDEX:
            continue
        box = match["box"]
        left = max(0, min(width, int(box["left"])))
        top = max(0, min(height, int(box["top"])))
        right = max(0, min(width, int(box["left"] + box["width"])))
        bottom = max(0, min(height, int(box["top"] + box["height"])))
        if right > left and bottom > top:
            mask[top:bottom, left:right] = CLASS_INDEX[class_name]
    return mask


def matches_by_page(doc: dict) -> dict[str, list[dict]]:
    """Group one score's matches by the page image they sit on."""
    grouped: dict[str, list[dict]] = {}
    for match in doc.get("matches", []):
        grouped.setdefault(match["page_image"], []).append(match)
    return grouped


def _find_page(score_id: str, page_image: str, pngs_dirs: list[Path]) -> Path | None:
    # OSSQ pages live under a deep per-work path (`<composer>/<work>/images/scanned/
    # original/`) rather than the Lieder corpus's flat `<pngs>/<score>/<page>`, so its
    # extractor records the absolute path and there is nothing to join.
    if page_image.startswith("/"):
        absolute = Path(page_image)
        return absolute if absolute.is_file() else None
    for pngs_dir in pngs_dirs:
        candidate = pngs_dir / score_id / page_image
        if candidate.is_file():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--matches", type=Path, required=True,
        help="ocr_first_text_ground_truth.py's --out dir.",
    )
    parser.add_argument("--pngs", type=Path, required=True, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--background-outside", action="store_true",
        help="Ablation: label unmatched pixels background instead of ignore.",
    )
    parser.add_argument(
        "--background-blank", action="store_true",
        help="Middle policy: unmatched blank-paper pixels are background, unmatched "
             "inked pixels stay ignore. Restores the negative supervision that pure "
             "ignore-masking gives up, without ever calling missed text background.",
    )
    parser.add_argument("--blank-threshold", type=int, default=BLANK_THRESHOLD)
    parser.add_argument("--score-ids", type=Path)
    args = parser.parse_args()

    wanted = None
    if args.score_ids:
        wanted = {s.strip() for s in args.score_ids.read_text().splitlines() if s.strip()}

    args.out.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    pages = skipped = 0
    class_pixels = {name: 0 for name in KIND_TO_CLASS.values()}
    supervised_pixels = 0
    total_pixels = 0

    for doc_path in sorted(args.matches.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        score_id = doc["score_id"]
        if wanted is not None and score_id not in wanted:
            continue
        for page_image, matches in matches_by_page(doc).items():
            page_path = _find_page(score_id, page_image, args.pngs)
            if page_path is None:
                skipped += 1
                continue
            image = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                skipped += 1
                continue
            mask = mask_for_page(
                image.shape[1], image.shape[0], matches, args.background_outside,
                image=image if args.background_blank else None,
                blank_threshold=args.blank_threshold,
            )
            mask_path = args.out / f"{score_id}_{Path(page_image).stem}.mask.png"
            cv2.imwrite(str(mask_path), mask)
            pairs.append((str(page_path), str(mask_path)))
            pages += 1
            total_pixels += mask.size
            supervised_pixels += int((mask != IGNORE).sum())
            for name, index in CLASS_INDEX.items():
                if name in class_pixels:
                    class_pixels[name] += int((mask == index).sum())

    index_path = args.out / "index.txt"
    index_path.write_text(
        "\n".join(f"{image},{mask}" for image, mask in pairs) + "\n", encoding="utf-8"
    )

    if args.background_outside:
        policy = "background-outside (ablation)"
    elif args.background_blank:
        policy = f"background-on-blank-paper (threshold {args.blank_threshold})"
    else:
        policy = "ignore-outside"
    print(f"{pages:,} page masks written, {skipped} skipped, policy: {policy}")
    if total_pixels:
        print(f"  supervised pixels: {supervised_pixels / total_pixels:.2%} of all pixels")
        for name, count in class_pixels.items():
            print(f"  {name}: {count / total_pixels:.4%} of all pixels")
    print(f"  index: {index_path}")


if __name__ == "__main__":
    main()
