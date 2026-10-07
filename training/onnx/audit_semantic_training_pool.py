"""Count remaining calibration positives without reading any formal test pages."""

import argparse
import json
from collections import Counter
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_receipt.read_text())
    excluded = set(selection["excluded_scores"]) | set(selection["selected_scores"])
    rows = [json.loads(line) for line in args.calibration.read_text().splitlines()]
    if any(row["partition"] != "calibration" for row in rows):
        raise ValueError("Require a calibration-only manifest")
    kept = [row for row in rows if row["corpus"] == "lieder" and row["score"] not in excluded]
    roles = Counter(g["semanticFamily"] for row in kept for g in row["ground_truth"])
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "report.json",
        dict(
            candidate_pages=len(kept),
            candidate_scores=len({r["score"] for r in kept}),
            positive_regions=roles,
            scores_by_positive_role={
                role: len(
                    {
                        row["score"]
                        for row in kept
                        if any(g["semanticFamily"] == role for g in row["ground_truth"])
                    }
                )
                for role in roles
            },
            excluded_scores=sorted(excluded),
            selected_page_ids=[r["id"] for r in kept],
            calibration_sha256=sha(args.calibration),
            selection_receipt_sha256=sha(args.selection_receipt),
            code_sha256=sha(__file__),
            limitation="Metadata-only audit. Partial positive annotations, not verified training labels "
            "or complete negatives. No formal test, OCR, classifier training or threshold tuning. "
            "No interpretation of unannotated text as non-lyric.",
        ),
    )
    print(
        json.dumps(dict(pages=len(kept), scores=len({r["score"] for r in kept}), positives=roles))
    )


if __name__ == "__main__":
    main()
