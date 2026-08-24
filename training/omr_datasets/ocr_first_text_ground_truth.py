"""OCR-first Stage 3 (lyrics/dynamics) text-region ground truth: OCR each matched
scan page directly, then search the OCR output for text already known to be
correct (from the piece's own real MusicXML) - the "OCR-first" approach scoped in
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §7, as opposed to proposing candidate
regions with the existing (partially broken) `detector_masks_v4` text detector
first. Chosen over that alternative: it doesn't depend on a detector whose
precision has already collapsed on several classes, and it's simpler - one pass,
no separate proposal step.

Reuses `rapidocr` (already a dependency, already used the same way by
`homr/title_detection.py` for exactly this kind of "read printed text off a real
scan" task) rather than adding a new OCR engine.

Scoping: rather than matching a page's OCR output against a piece's *entire*
lyrics list (risking a false match against a different page's own lyrics),
`page_measure_ranges` reuses `fetch_lieder_ground_truth.py`'s own per-page
per-system measure counts - already computed, already validated - to know which
measure range belongs to which scan page, and only searches the lyrics/dynamics
that actually fall in that range.
"""

# flake8: noqa: T201

import argparse
import difflib
import json
import re
from pathlib import Path

import yaml

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.musicxml_text_ground_truth import (
    extract_expected_texts,
    unzip_mxl,
    words_from_syllables,
)

_STRIP_PUNCTUATION_RE = re.compile(r"^[.,;:!?'\"()\[\]]+|[.,;:!?'\"()\[\]]+$")

#: A word/dynamic token counts as matched against an OCR token if their normalized
#: similarity clears this - loose enough for real OCR error on historical engraved
#: text, tight enough not to match unrelated short words by chance.
TOKEN_MATCH_THRESHOLD = 0.8
#: An OCR line is confirmed as containing lyrics if at least this fraction of its
#: own tokens match some expected word - not every token needs to match (OCR drops
#: or mangles some), but most should.
LINE_MATCH_THRESHOLD = 0.6


def page_measure_ranges(ground_truth_pages: list[list[int]]) -> list[tuple[int, int]]:
    """`(start, end)` measure-index range (half-open, matching Python slicing) per
    page, from `fetch_lieder_ground_truth.py`'s own per-page list of per-system
    measure counts - cumulative sum, same positional convention as everywhere else
    in this corpus (0-based, page/system order, not MusicXML's own `number`
    attribute)."""
    ranges = []
    cursor = 0
    for page in ground_truth_pages:
        total = sum(page)
        ranges.append((cursor, cursor + total))
        cursor += total
    return ranges


def _normalize_token(token: str) -> str:
    return _STRIP_PUNCTUATION_RE.sub("", token).lower()


def ocr_page(reader: object, image_path: Path) -> list[dict]:
    """`[{"box": {left,top,width,height}, "text": str, "score": float}, ...]` for
    one page, converting RapidOCR's own 4-point polygon boxes to the plain
    axis-aligned shape every other box in this corpus already uses."""
    result = reader(str(image_path))
    if result.boxes is None or result.txts is None:
        return []
    lines = []
    for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(
            {
                "box": {
                    "left": int(round(min(xs))),
                    "top": int(round(min(ys))),
                    "width": int(round(max(xs) - min(xs))),
                    "height": int(round(max(ys) - min(ys))),
                },
                "text": text,
                "score": float(score),
            }
        )
    return lines


def match_lyrics_to_ocr(
    expected_words: list[str], ocr_lines: list[dict], threshold: float = LINE_MATCH_THRESHOLD
) -> list[dict]:
    """OCR lines confirmed as containing lyrics - localization is per *OCR line*,
    not per word: RapidOCR's own text detection groups words into printed lines,
    and a lyric line under a system is exactly one such OCR line, not several -
    word-level boxes aren't available without a different OCR mode.
    """
    normalized_expected = {_normalize_token(w) for w in expected_words if _normalize_token(w)}
    confirmed = []
    for line in ocr_lines:
        tokens = line["text"].split()
        if not tokens:
            continue
        matched = sum(
            1
            for token in tokens
            if any(
                difflib.SequenceMatcher(None, _normalize_token(token), w).ratio()
                >= TOKEN_MATCH_THRESHOLD
                for w in normalized_expected
            )
        )
        fraction = matched / len(tokens)
        if fraction >= threshold:
            confirmed.append({**line, "kind": "lyric", "matched_fraction": fraction})
    return confirmed


def match_dynamics_to_ocr(expected_dynamics: list[str], ocr_lines: list[dict]) -> list[dict]:
    """OCR lines confirmed as one specific dynamics marking - dynamics are typeset
    as their own short, standalone mark (`p`, `f`, `cresc.`), not part of a longer
    text line, so this matches an OCR line's *whole* text against one expected
    mark, not token-by-token like `match_lyrics_to_ocr`.
    """
    normalized_expected = {_normalize_token(d) for d in expected_dynamics if _normalize_token(d)}
    confirmed = []
    for line in ocr_lines:
        text_norm = _normalize_token(line["text"])
        if not text_norm:
            continue
        best_ratio = max(
            (difflib.SequenceMatcher(None, text_norm, d).ratio() for d in normalized_expected),
            default=0.0,
        )
        if best_ratio >= TOKEN_MATCH_THRESHOLD:
            confirmed.append({**line, "kind": "dynamic", "match_ratio": best_ratio})
    return confirmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--score-ids", type=Path, required=True, help="Text file, one score id per line."
    )
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="fetch_lieder_ground_truth.py's --out dir - for page/measure ranges.",
    )
    parser.add_argument("--systems", type=Path, required=True, help="imslp_systems(_repaired) dir.")
    parser.add_argument("--pngs", type=Path, required=True, help="Matching imslp_pngs dir.")
    parser.add_argument("--out", type=Path, required=True, help="Output dir for per-score JSON.")
    args = parser.parse_args()

    from rapidocr import RapidOCR  # deferred - a real, if modest, model-load cost

    reader = RapidOCR()

    lieder = load_lieder_scores(args.scores_yaml_cache)
    mscx_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)
    score_ids = [line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()]
    matched = match_single_piece_scores(lieder, score_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0
    for score_id, (key, entry) in matched.items():
        out_path = args.out / f"{score_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        gt_path = args.ground_truth / f"{score_id}.json"
        systems_path = args.systems / f"{score_id}.yaml"
        if not gt_path.exists() or not systems_path.exists():
            print(f"{score_id}: missing ground truth or detected systems, skipping")
            continue

        try:
            ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
            systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
            mxl_bytes = fetch_mxl(entry, key, mxl_tree or mscx_tree)
            musicxml_bytes = unzip_mxl(mxl_bytes)
            expected = extract_expected_texts(musicxml_bytes)
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED preparing ground truth ({e})")
            failed += 1
            continue

        ranges = page_measure_ranges(ground_truth["pages"])
        detected_pages = [systems_doc["pages"][k] for k in sorted(systems_doc["pages"])]

        matches: list[dict] = []
        for page_index, (start, end) in enumerate(ranges):
            if page_index >= len(detected_pages):
                break
            page_words = words_from_syllables(
                [e for e in expected if e["kind"] == "lyric" and start <= e["measure_index"] < end]
            )
            page_dynamics = [
                e["text"] for e in expected
                if e["kind"] == "dynamic" and start <= e["measure_index"] < end
            ]
            if not page_words and not page_dynamics:
                continue

            page_path = args.pngs / detected_pages[page_index]["image"]
            try:
                ocr_lines = ocr_page(reader, page_path)
            except Exception as e:  # noqa: BLE001
                print(f"{score_id} page {page_index}: OCR FAILED ({e})")
                continue

            for m in match_lyrics_to_ocr(page_words, ocr_lines):
                matches.append({**m, "page_index": page_index, "page_image": str(page_path.name)})
            for m in match_dynamics_to_ocr(page_dynamics, ocr_lines):
                matches.append({**m, "page_index": page_index, "page_image": str(page_path.name)})

        out_path.write_text(
            json.dumps({"score_id": score_id, "matches": matches}, indent=2), encoding="utf-8"
        )
        lyric_count = sum(1 for m in matches if m["kind"] == "lyric")
        dynamic_count = sum(1 for m in matches if m["kind"] == "dynamic")
        print(f"{score_id}: {lyric_count} lyric line(s), {dynamic_count} dynamic mark(s) confirmed")
        ok += 1

    print(f"{ok} processed, {skipped} already cached, {failed} failed")


if __name__ == "__main__":
    main()
