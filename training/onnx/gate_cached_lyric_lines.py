"""Select intact cached OCR regions by pinned Lyrics-box overlap; no reference input."""

import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def coverage(box, supports):
    xs, ys = [p[0] for p in box], [p[1] for p in box]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    area = (right - left) * (bottom - top)
    if area <= 0:
        raise ValueError("Empty OCR rectangle")
    clipped = []
    for row in supports:
        x0, y0, x1, y1 = (
            max(left, row["left"]),
            max(top, row["top"]),
            min(right, row["right"]),
            min(bottom, row["bottom"]),
        )
        if x1 > x0 and y1 > y0:
            clipped.append((x0, y0, x1, y1))
    edges = sorted({x for r in clipped for x in (r[0], r[2])})
    union = 0.0
    for x0, x1 in zip(edges, edges[1:]):
        spans = sorted((r[1], r[3]) for r in clipped if r[0] <= x0 and r[2] >= x1)
        length = 0.0
        end = None
        for a, b in spans:
            length += max(0, b - max(a, end if end is not None else a))
            end = max(end if end is not None else b, b)
        union += (x1 - x0) * length
    return union / area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    pages = json.loads(args.pages.read_text())
    if any(p["partition"] != "calibration" for p in pages):
        raise ValueError("Calibration only")
    args.out.mkdir(parents=True)
    summary = []
    for area in (4, 200):
        for threshold in (0.1, 0.25, 0.5):
            directory = args.out / f"area{area}-coverage{threshold:g}"
            directory.mkdir()
            for page in pages:
                ocr_path = args.ocr / (page["score"] + ".json")
                boxes_path = args.regions / (page["score"] + ".json")
                original = json.loads(ocr_path.read_text())
                regions = json.loads(boxes_path.read_text())
                if (
                    original["image_sha256"] != page["image_sha256"]
                    or regions["image_sha256"] != page["image_sha256"]
                ):
                    raise ValueError("Image identities differ")
                supports = regions["areas"][str(area)]
                selected = []
                rejected = []
                for item in original["all_page_text"]:
                    overlap = coverage(item["box"], supports)
                    row = dict(item, lyrics_rectangle_union_coverage=overlap)
                    (selected if overlap >= threshold else rejected).append(row)
                record = dict(
                    original,
                    all_page_text=selected,
                    rejected_page_text=rejected,
                    selection={"min_area": area, "minimum_union_coverage": threshold},
                    original_ocr_sha256=sha(ocr_path),
                    detector_regions_sha256=sha(boxes_path),
                    source_text_or_reference_geometry_used=False,
                )
                write(directory / (page["score"] + ".json"), record)
                summary.append(
                    {
                        "variant": directory.name,
                        "score": page["score"],
                        "selected": len(selected),
                        "rejected": len(rejected),
                    }
                )
    write(
        args.out / "receipt.json",
        {
            "source_sha256": sha(Path(__file__)),
            "pages_sha256": sha(args.pages),
            "ocr_receipt_sha256": sha(args.ocr / "receipt.json"),
            "detector_receipt_sha256": sha(args.regions / "receipt.json"),
            "summary": summary,
            "scope": (
                "Predeclared overlap-grid calibration diagnostic; selection uses no lyrics text, "
                "source or manual rows; no attachment"
            ),
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
