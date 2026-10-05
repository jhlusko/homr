"""Try OCR line detection inside fixed pinned-e2 boxes; no new detector fitting."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    pages = json.loads(args.pages.read_text())
    if any(p["partition"] != "calibration" for p in pages):
        raise ValueError("Calibration only")
    receipt = json.loads((args.regions / "receipt.json").read_text())
    if receipt["pages_sha256"] != sha(args.pages):
        raise ValueError("Region pages differ")
    ocr = receipt["ocr_identity"]
    import cv2
    from rapidocr import RapidOCR

    if importlib.metadata.version("rapidocr") != ocr["ocr_version"]:
        raise ValueError("OCR package changed")
    for m in ocr["models"].values():
        if sha(m["path"]) != m["sha256"]:
            raise ValueError("OCR model changed")
    cv2.setNumThreads(1)
    engine = RapidOCR(params=ocr["parameters"])
    args.out.mkdir(parents=True)
    inputs = {}
    summary = []
    for page in pages:
        path = args.regions / (page["score"] + ".json")
        inputs[page["score"]] = sha(path)
        original = json.loads(path.read_text())
        if original["image_sha256"] != sha(page["image"]):
            raise ValueError("Original image changed")
        image = cv2.imread(page["image"])
        cache = {}
        areas = {}
        tick = time.monotonic()
        for area, boxes in original["areas"].items():
            rows = []
            for box in boxes:
                key = tuple(box[n] for n in ("left", "top", "right", "bottom"))
                if key not in cache:
                    left, top, right, bottom = key
                    result = engine(image[top:bottom, left:right], use_det=True)
                    cache[key] = []
                    if result.txts is not None:
                        for coords, text, confidence in zip(
                            result.boxes, result.txts, result.scores, strict=True
                        ):
                            xs = [float(p[0]) + left for p in coords]
                            ys = [float(p[1]) + top for p in coords]
                            cache[key].append(
                                {
                                    "label": "Lyrics",
                                    "left": min(xs),
                                    "top": min(ys),
                                    "right": max(xs),
                                    "bottom": max(ys),
                                    "text": text,
                                    "ocr_confidence": float(confidence),
                                    "parent_detector_box": key,
                                }
                            )
                rows.extend(cache[key])
            areas[area] = rows
        write(
            args.out / (page["score"] + ".json"),
            {
                "score": page["score"],
                "page_id": page["id"],
                "image_sha256": page["image_sha256"],
                "areas": areas,
                "source_text_used": False,
            },
        )
        summary.append(
            {
                "score": page["score"],
                "seconds": time.monotonic() - tick,
                "regions": {a: len(v) for a, v in areas.items()},
            }
        )
        print(json.dumps(summary[-1]), flush=True)
    write(
        args.out / "receipt.json",
        {
            "pages_sha256": sha(args.pages),
            "parent_regions_sha256": inputs,
            "parent_receipt_sha256": sha(args.regions / "receipt.json"),
            "source_sha256": sha(Path(__file__)),
            "ocr_identity": ocr,
            "detector": receipt["detector"],
            "results": summary,
            "recognition_use_det": True,
            "scope": "Calibration diagnostic inside fixed detector regions; no attachment",
        },
    )


if __name__ == "__main__":
    main()
