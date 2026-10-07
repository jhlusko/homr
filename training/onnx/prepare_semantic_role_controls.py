"""Freeze previously unused C1 calibration pages; annotations stay evaluation-only."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write

RESERVED = {"IMSLP154086", "IMSLP89797"}


def select(rows, excluded):
    dual, directions = [], []
    for row in rows:
        if (
            row["partition"] != "calibration"
            or row["corpus"] != "lieder"
            or row["score"] in excluded
        ):
            continue
        counts = Counter(g["semanticFamily"] for g in row["ground_truth"])
        if counts["Lyrics"] >= 4 and counts["DirectionText"] >= 1:
            dual.append(row)
        elif counts["DirectionText"] >= 2:
            directions.append(row)

    def order(row):
        return hashlib.sha256(("20261006" + row["id"]).encode()).hexdigest()

    chosen, seen = [], set()
    for candidates, limit in ((sorted(dual, key=order), 5), (sorted(directions, key=order), 3)):
        count = 0
        for row in candidates:
            if row["score"] in seen:
                continue
            chosen.append(row)
            seen.add(row["score"])
            count += 1
            if count == limit:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--previous-pages", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    excluded = set(RESERVED)
    for path in args.previous_pages:
        excluded.update(r["score"] for r in json.loads(path.read_text()))
    chosen = select(
        [json.loads(line) for line in args.calibration.read_text().splitlines()], excluded
    )
    if len(chosen) != 8:
        raise ValueError("Expected five dual-role and three direction-rich pages")
    for page in chosen:
        if sha(Path(page["image"])) != page["image_sha256"]:
            raise ValueError("Selected image changed")
        for annotation in page["annotations"]:
            if sha(Path(annotation["path"])) != annotation["sha256"]:
                raise ValueError("Selected annotation changed")
    args.out.mkdir(parents=True, exist_ok=False)
    inference = [
        {k: r[k] for k in ("id", "score", "partition", "image", "image_sha256")} for r in chosen
    ]
    write(args.out / "inference-pages.json", inference)
    write(args.out / "evaluation-pages.json", chosen)
    write(
        args.out / "receipt.json",
        dict(
            calibration_manifest_sha256=sha(args.calibration),
            previous_pages_sha256={str(p): sha(p) for p in args.previous_pages},
            excluded_scores=sorted(excluded),
            selected_scores=[r["score"] for r in chosen],
            source_sha256=sha(Path(__file__)),
            inference_receives_annotations=False,
            formal_test_read=False,
            selection="Five pages with >=4 existing lyric positives and a direction; "
            "three additional pages with >=2 direction positives; score-disjoint, seeded pixel-ID ordering.",
            limitation="Unused in listed C1 calibration runs, not guaranteed unseen by detectors or core. "
            "Existing annotations are positive-only evidence, not complete semantic negatives. "
            "Original scans must confirm candidate roles; unannotated OCR remains unknown.",
        ),
    )


if __name__ == "__main__":
    main()
