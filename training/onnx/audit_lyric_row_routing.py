"""Route preselected scan OCR rows; score reviewed staff routing separately."""

import argparse
import json
from pathlib import Path

from training.onnx.lyric_row_routing import route_rows
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_grouped_lyric_positions import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages, hashes = [], {}
    for page_index, score in enumerate(("IMSLP154060", "IMSLP89794")):
        transform_path = args.geometry / f"page-{page_index}/staff-transforms.json"
        selected_path = args.selected / f"{score}.json"
        position_path = args.positions / f"{score}.json"
        for path in (transform_path, selected_path, position_path):
            hashes[str(path)] = sha(path)
        selected = json.loads(selected_path.read_text())
        positions = json.loads(position_path.read_text())
        transforms = json.loads(transform_path.read_text())
        rows = selected["all_page_text"]
        result = route_rows(rows, transforms)
        # Frozen reviewed references enter evaluation only, after source-free routing.
        references = (
            [(box, system, verse) for box, system, verse, _ in CASES]
            if page_index == 0
            else [(58, 2, "1")]
        )
        reviewed = []
        for box_index, expected_call, native_verse in references:
            box = positions["rows"][box_index]["box"]
            indices = [i for i, row in enumerate(rows) if row["box"] == box]
            if len(indices) != 1:
                reviewed.append({"ocr_box_index": box_index, "selection_unresolved": True})
                continue
            routed = [r for r in result["routed"] if r["index"] == indices[0]]
            reviewed.append(
                {
                    "ocr_box_index": box_index,
                    "expected_staff_call": expected_call,
                    "native_reference_verse": native_verse,
                    "prediction": routed[0] if routed else None,
                    "staff_correct": bool(routed) and routed[0]["staff_call"] == expected_call,
                }
            )
        pages.append(
            {
                "score": score,
                "selected_rows": len(rows),
                **result,
                "reviewed_staff_checks": reviewed,
                "rows": rows,
            }
        )
    report = {
        "scope": (
            "Two calibration scans, cached source-independent OCR selection; reviewed staff checks only, no native verse prediction"
        ),
        "source_used_by_router": False,
        "production_qualified": False,
        "pages": pages,
        "input_sha256": hashes,
        "code_sha256": {
            str(path): sha(path)
            for path in (Path(__file__), Path(__file__).with_name("lyric_row_routing.py"))
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, report)
    for page in pages:
        print(
            page["score"],
            "selected",
            page["selected_rows"],
            "routed",
            len(page["routed"]),
            "unresolved",
            len(page["unresolved"]),
            "reviewed_correct",
            sum(r.get("staff_correct", False) for r in page["reviewed_staff_checks"]),
        )


if __name__ == "__main__":
    main()
