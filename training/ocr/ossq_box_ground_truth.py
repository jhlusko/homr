"""Turn the OSSQ OCR-first text matches into box ground truth `detector_box_eval` reads.

The instrumental detector has only ever been measured on patch IoU, and this project has
already established that patch IoU does not predict page-level behaviour: E3 led the
patch table on real-scan Lyrics (0.966) and came *last* of six on page-level F1. Shipping
the instrumental model on patch numbers alone would repeat exactly that mistake.

`detector_box_eval.py` needs `<dir>/<score>/<page>.boxes.json` beside the page image.
The OCR-first extraction already produced boxes with classes - 36,897 of them across 93
scores - so this is a format conversion, not new labelling.

**The one thing that must be understood before reading any number this produces.** These
boxes are *OCR-confirmed* text: a box exists only where RapidOCR read something *and* it
matched the score's own MuseScore text. Real text the OCR missed is absent from the
ground truth, so:

- **Recall is trustworthy.** Of the marks we know are on the page, how many did the
  detector find? Nothing about a missing box inflates that.
- **Precision is a lower bound, not a measurement.** A detector that correctly finds a
  dynamic the OCR missed is counted as a false positive, because the ground truth does
  not know that mark exists.

This is the evaluation-time face of the same asymmetry `scan_text_masks.py` handles at
training time with `ignore_index`: incomplete positives are safe to learn from and unsafe
to be scored against. A precision figure from this data should be quoted as "at least",
and a *comparison* between two detectors on the same incomplete ground truth is more
meaningful than either absolute number.
"""

# flake8: noqa: T201

import argparse
import json
from collections import defaultdict
from pathlib import Path

#: OCR-first `kind` -> detector class, matching `scan_text_masks.KIND_TO_CLASS`.
KIND_TO_LABEL = {
    "lyric": "Lyrics",
    "dynamic": "Dynamic",
    "tempo": "Tempo",
    "stafftext": "StaffText",
    "expression": "Expression",
}


def record_for(matches: list[dict], image_name: str) -> dict:
    """One page's `.boxes.json`, in `detector_data.boxes_of`'s shape.

    Lyrics go in their own key because that is where `boxes_of` looks for them; every
    other class goes under `text_boxes`. An instrumental corpus produces no lyrics at
    all, and the empty list is written anyway so the file shape never varies.
    """
    text_boxes: dict[str, list[dict]] = defaultdict(list)
    lyrics: list[dict] = []
    for match in matches:
        label = KIND_TO_LABEL.get(match.get("kind", ""))
        if label is None:
            continue
        box = match["box"]
        entry = {
            "left": int(box["left"]),
            "top": int(box["top"]),
            "right": int(box["left"] + box["width"]),
            "bottom": int(box["top"] + box["height"]),
        }
        if label == "Lyrics":
            lyrics.append(entry)
        else:
            text_boxes[label].append(entry)
    return {"image": image_name, "lyrics": lyrics, "text_boxes": dict(text_boxes)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--score-ids", type=Path,
        help="Restrict to these scores, e.g. a validation split's own ids.",
    )
    parser.add_argument(
        "--index-out", type=Path,
        help="Write an index naming the symlinked pages. Required in practice: "
             "`detector_box_eval` keys ground truth on the path recorded in the "
             "boxes.json (which resolves next to it) and looks it up by the path in "
             "the index. Handing it the corpus's own index instead pairs two different "
             "spellings of the same page, matches nothing, and reports 0.0% across "
             "every class with `gt boxes = 0` - a total failure that looks like a "
             "uselessly bad model.",
    )
    args = parser.parse_args()

    wanted = None
    if args.score_ids:
        wanted = {s.strip() for s in args.score_ids.read_text().splitlines() if s.strip()}

    args.out.mkdir(parents=True, exist_ok=True)
    pages = boxes = 0
    index_rows: list[str] = []
    for doc_path in sorted(args.matches.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        score_id = doc["score_id"]
        if wanted is not None and score_id not in wanted:
            continue
        by_page: dict[str, list[dict]] = defaultdict(list)
        for match in doc.get("matches", []):
            by_page[match["page_image"]].append(match)

        score_dir = args.out / score_id
        score_dir.mkdir(parents=True, exist_ok=True)
        for page_path, matches in by_page.items():
            source = Path(page_path)
            if not source.is_file():
                continue
            # The evaluator resolves the image relative to the json, and OSSQ pages live
            # under deep per-work paths. A symlink keeps one copy of a 12MP scan.
            link = score_dir / source.name
            if not link.exists():
                link.symlink_to(source)
            record = record_for(matches, source.name)
            (score_dir / f"{source.stem}.boxes.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            pages += 1
            boxes += len(record["lyrics"]) + sum(len(v) for v in record["text_boxes"].values())
            index_rows.append(f"{link},{score_dir / f'{source.stem}.boxes.json'}")

    if args.index_out:
        args.index_out.write_text("\n".join(index_rows) + "\n", encoding="utf-8")
        print(f"index of {len(index_rows):,} pages -> {args.index_out}")

    print(f"{pages:,} pages, {boxes:,} boxes -> {args.out}")
    print("NOTE: OCR-confirmed boxes only. Recall is sound; precision is a lower bound.")


if __name__ == "__main__":
    main()
