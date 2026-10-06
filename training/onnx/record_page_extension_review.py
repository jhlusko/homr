"""Record a human page-boundary review against a frozen extension selection."""

import argparse
import json
import os
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review-page", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite review evidence")
    selection = json.loads(args.selection.read_text())
    scores = {}
    for row in selection:
        counts = [s["end_measure"] - s["start_measure"] for s in row["systems"]]
        artifacts = [
            args.review_page,
            *[
                args.selection.parent / row["score"] / name
                for name in ("source-preview.pdf", "source-full.png")
            ],
        ]
        if (
            sha(row["image"]) != row["image_sha256"]
            or sha(row["reference"]) != row["reference_sha256"]
        ):
            raise ValueError("Reviewed input changed")
        scores[row["score"]] = {
            "edition_identity_checked": True,
            "complete_page_checked": True,
            "part_order_checked": True,
            "reference_sha256": row["reference_sha256"],
            "image_sha256": row["image_sha256"],
            "visible_system_measure_counts": counts,
            "source_parts": len(read_reference(Path(row["reference"])).findall("part")),
            "boundary_evidence": (
                ["human review of scan against canonical source metadata"] * len(counts)
            ),
            "review_artifacts": {
                os.path.relpath(p, args.selection.parent): sha(p) for p in artifacts
            },
        }
    write(
        args.out,
        {
            "selection_sha256": sha(args.selection),
            "reviewer": args.reviewer,
            "confirmation": args.confirmation,
            "tool_sha256": sha(Path(__file__)),
            "model_predictions_used": False,
            "scores": scores,
        },
    )


if __name__ == "__main__":
    main()
