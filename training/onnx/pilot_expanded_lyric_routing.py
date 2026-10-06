"""Capture segmentation staff geometry and route cached OCR without core decoding."""

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from training.onnx.lyric_row_routing import route_rows
from training.onnx.qualify_joint_export import setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    setup()
    from homr import main as pipeline
    from homr.segmentation.config import segnet_path_onnx

    pages = json.loads(args.pages.read_text())
    if any(page["partition"] != "calibration" for page in pages):
        raise ValueError("Calibration pages only")
    root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__),
        Path(__file__).with_name("lyric_row_routing.py"),
        root / "homr/main.py",
        root / "homr/staff_detection.py",
    ]
    for path in sources:
        target = args.out / "source-snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    hashes = {str(path): sha(path) for path in sources}
    write(
        args.out / "identity.json",
        {
            "pages_sha256": sha(args.pages),
            "code_sha256": hashes,
            "segmentation": {"path": segnet_path_onnx, "sha256": sha(segnet_path_onnx)},
            "threads": 1,
            "core_decoding": False,
            "source_used_by_inference": False,
            "staff_ids": "Physical pre-merge staff indices; not runtime voice or native part IDs",
        },
    )
    original_merge = pipeline.find_braces_brackets_and_grand_staff_lines
    original_crop = pipeline.autocrop
    captured = []

    def capture_merge(debug, staffs, braces):
        for index, staff in enumerate(staffs):
            captured.append(
                {
                    "staff_call": index,
                    "staff_grid": [{"x": float(p.x), "y": list(p.y)} for p in staff.grid],
                }
            )
        return original_merge(debug, staffs, braces)

    def require_identity_crop(image):
        cropped = original_crop(image)
        if cropped.shape != image.shape or not np.array_equal(cropped, image):
            raise ValueError("Nonidentity autocrop: original-page offset is not captured")
        return cropped

    pipeline.find_braces_brackets_and_grand_staff_lines = capture_merge
    pipeline.autocrop = require_identity_crop
    summaries = []
    for index, page in enumerate(pages):
        folder = args.out / f"page-{index}"
        folder.mkdir()
        captured.clear()
        started = time.monotonic()
        write(args.out / "progress.json", {"active": page["score"], "completed": summaries})
        record = {"score": page["score"], "page_index": index}
        try:
            if sha(page["image"]) != page["image_sha256"]:
                raise ValueError("Image changed")
            selected_path = args.selected / f"{page['score']}.json"
            selected = json.loads(selected_path.read_text())
            if selected["image_sha256"] != page["image_sha256"]:
                raise ValueError("OCR selection image differs")
            image_path = folder / ("page" + Path(page["image"]).suffix)
            shutil.copyfile(page["image"], image_path)
            original_size = np.array(cv2.imread(str(image_path)).shape[:2][::-1])
            _, image, _, _, _ = pipeline.detect_staffs_in_image(
                str(image_path),
                pipeline.ProcessingConfig(
                    False, False, False, False, -1, False, False, False, False
                ),
            )
            ratio = (original_size / np.array(image.shape[:2][::-1])).tolist()
            for staff in captured:
                staff["original_to_processed_ratio"] = ratio
            write(folder / "staff-transforms.json", captured)
            rows = selected["all_page_text"]
            result = route_rows(rows, captured)
            write(
                folder / "routing.json",
                {
                    "page": page,
                    "rows": rows,
                    **result,
                    "selection_sha256": sha(selected_path),
                    "scope": (
                        "Physical staff-gap coverage only; no native verse, part or note ownership qualification"
                    ),
                },
            )
            record.update(
                status="ok",
                physical_staffs=len(captured),
                selected=len(rows),
                routed=len(result["routed"]),
                unresolved=len(result["unresolved"]),
            )
        except Exception:
            record.update(status="failed", error=traceback.format_exc())
        record["seconds"] = time.monotonic() - started
        write(folder / "result.json", record)
        summaries.append(record)
    write(args.out / "progress.json", {"active": None, "completed": summaries})
    write(
        args.out / "completion.json",
        {"pages": summaries, "success": all(row["status"] == "ok" for row in summaries)},
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
