"""OCR-first text ground truth for the OSSQ-OMR instrumental corpus.

The Lieder extractor (`ocr_first_text_ground_truth.py`) is built around that corpus's
shape: an IMSLP-to-Lieder mapping, cached GitHub file trees, per-system crops, and
measure ranges used to decide which lyrics belong on which system. None of that exists
here, and none of it is needed - OSSQ ships its own scanned pages next to the MuseScore
file they were engraved from.

What is shared is the part that matters: `ocr_page` reads a page, and
`match_dynamics_to_ocr` confirms an OCR line against a set of expected printed strings.
Dynamics, tempo marks, staff text and expression markings are all typeset the same way -
short, standalone, on their own line - so one rule covers all four. Lyrics are the
exception that needed token-level matching, and there are none in a string quartet.

**Why this corpus is worth extracting at all.** Every scan-derived training example so
far comes from Lieder, which is vocal music: 83% of its boxes are lyrics. A detector
meant for instrumental scores has never seen a real instrumental scan, and the classes
that carry instrumental scores - Tempo, StaffText, Expression - had no real-scan
supervision whatsoever. Measured over the 96 OSSQ scores that have both a `.mscx` and
scanned pages, the MuseScore sources carry 130,981 dynamics, 15,631 tempo marks, 8,732
staff texts and 3,804 expression markings.

Output matches the Lieder extractor's JSON exactly, so `scan_text_masks.py` and
everything downstream of it work unchanged. Page paths are absolute because OSSQ pages
live under a deep per-work directory rather than a flat `<pngs>/<score>/` layout.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

from training.omr_datasets.mscx_text_ground_truth import load_mscx, texts_by_kind
from training.omr_datasets.ocr_first_text_ground_truth import (
    match_dynamics_to_ocr,
    ocr_page,
)

#: Every kind the `.mscx` yields is matched by the standalone-mark rule.
MATCHED_KINDS = ("dynamic", "tempo", "stafftext", "expression")


def score_dirs(root: Path) -> list[Path]:
    """Work directories holding both a MuseScore source and scanned pages.

    A score with one and not the other cannot produce a training pair, so it is
    skipped here rather than failing later with a confusing missing-file error.
    """
    found = []
    for mscx in sorted(root.glob("*/*/sq*.mscx")):
        if (mscx.parent / "images" / "scanned" / "original").is_dir():
            found.append(mscx.parent)
    return found


def score_id_of(work_dir: Path) -> str:
    mscx = next(iter(sorted(work_dir.glob("sq*.mscx"))))
    return mscx.stem


def pages_of(work_dir: Path) -> list[Path]:
    return sorted((work_dir / "images" / "scanned" / "original").glob("*.png"))


def matches_for_score(reader: object, work_dir: Path) -> list[dict]:
    """Every confirmed text box across one score's scanned pages."""
    expected = texts_by_kind(load_mscx(next(iter(sorted(work_dir.glob("sq*.mscx"))))))
    confirmed: list[dict] = []
    for page in pages_of(work_dir):
        lines = ocr_page(reader, page)
        if not lines:
            continue
        claimed: set[tuple] = set()
        for kind in MATCHED_KINDS:
            wanted = expected.get(kind)
            if not wanted:
                continue
            for match in match_dynamics_to_ocr(wanted, lines, kind=kind):
                box = match["box"]
                key = (box["left"], box["top"], box["width"], box["height"])
                # One printed mark is one box. Without this, a string that appears in
                # two kinds' expected sets - "cresc." is written as both an expression
                # and a staff text in this corpus - would emit the same pixels twice
                # under two different classes, and the mask would keep whichever was
                # rasterised last.
                if key in claimed:
                    continue
                claimed.add(key)
                confirmed.append({**match, "page_image": str(page)})
    return confirmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--ocr-threads", type=int, default=4)
    args = parser.parse_args()

    from rapidocr import RapidOCR  # deferred - a real, if modest, model-load cost

    reader = RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": args.ocr_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": args.ocr_threads,
        }
    )

    args.out.mkdir(parents=True, exist_ok=True)
    works = score_dirs(args.scores_root)
    mine = [w for i, w in enumerate(works) if i % args.shards == args.shard]
    print(f"{len(works)} scores, {len(mine)} in shard {args.shard}/{args.shards}", flush=True)

    for work_dir in mine:
        score_id = score_id_of(work_dir)
        out_path = args.out / f"{score_id}.json"
        if out_path.exists():
            print(f"{score_id}: already done", flush=True)
            continue
        try:
            matches = matches_for_score(reader, work_dir)
        except Exception as e:  # noqa: BLE001 - one bad score must not end the shard
            print(f"{score_id}: FAILED {e}", flush=True)
            continue
        out_path.write_text(
            json.dumps({"score_id": score_id, "matches": matches}, indent=1), encoding="utf-8"
        )
        print(f"{score_id}: {len(matches)} match(es) over {len(pages_of(work_dir))} page(s)",
              flush=True)


if __name__ == "__main__":
    main()
