"""Calibration-only pinned-e2 region OCR; compare area4 and proposed area200."""

import argparse
import importlib.metadata
import json
import time
from dataclasses import asdict
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ocr-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = json.loads(args.pages.read_text())
    if any(p["partition"] != "calibration" for p in pages) or args.out.exists():
        raise ValueError("Require calibration-only pages and new output")
    import cv2
    import torch
    from detector_runtime.detector_inference import boxes_from_probs, predict_mask
    from detector_runtime.runtime import TextDetector
    from pins import DETECTORS
    from rapidocr import RapidOCR

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    repo, revision, filename, digest = DETECTORS["lyrics"]
    detector = TextDetector(
        detector_id="lyrics",
        checkpoint=args.weights,
        revision=revision,
        sha256=digest,
        device="cpu",
    )
    ocr_identity = json.loads(args.ocr_receipt.read_text())
    if importlib.metadata.version("rapidocr") != ocr_identity["ocr_version"]:
        raise ValueError("OCR package differs")
    for model in ocr_identity["models"].values():
        if sha(model["path"]) != model["sha256"]:
            raise ValueError("OCR model differs")
    engine = RapidOCR(params=ocr_identity["parameters"])
    args.out.mkdir(parents=True)
    results = []
    for page in pages:
        if sha(page["image"]) != page["image_sha256"]:
            raise ValueError("Original page changed")
        image = cv2.imread(page["image"])
        tick = time.monotonic()
        probs = predict_mask(detector.model, image, "cpu", batch_size=2)
        areas = {}
        cache = {}
        for area in (4, 200):
            rows = []
            for box in boxes_from_probs(probs, min_area=area):
                if box.label != "Lyrics":
                    continue
                key = (box.left, box.top, box.right, box.bottom)
                if key not in cache:
                    result = engine(
                        image[box.top : box.bottom, box.left : box.right], use_det=False
                    )
                    cache[key] = {
                        "text": " ".join(result.txts or ()),
                        "ocr_confidence": list(result.scores) if result.scores is not None else [],
                    }
                rows.append(dict(asdict(box), **cache[key]))
            areas[str(area)] = rows
        record = {
            "score": page["score"],
            "page_id": page["id"],
            "image_sha256": page["image_sha256"],
            "seconds": time.monotonic() - tick,
            "areas": areas,
            "source_text_used": False,
        }
        write(args.out / (page["score"] + ".json"), record)
        results.append(
            {
                "score": page["score"],
                "seconds": record["seconds"],
                "regions": {a: len(v) for a, v in areas.items()},
            }
        )
        print(json.dumps(results[-1]), flush=True)
    write(
        args.out / "receipt.json",
        {
            "source_sha256": sha(Path(__file__)),
            "pages_sha256": sha(args.pages),
            "detector": {
                "repository": repo,
                "revision": revision,
                "sha256": digest,
                "path": str(args.weights),
                "filename": filename,
            },
            "ocr_identity": ocr_identity,
            "areas": [4, 200],
            "recognition_use_det": False,
            "results": results,
            "scope": (
                "Pinned-e2 calibration diagnostic; proposed area200 remains unfrozen; "
                "no note attachment"
            ),
        },
    )


if __name__ == "__main__":
    main()
