"""Unprompted full-scan OCR on linked calibration pages; no attachment score."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    pages = json.loads(args.pages.read_text())
    if any(p["partition"] != "calibration" for p in pages):
        raise ValueError("Only calibration pages are permitted")
    import rapidocr
    from rapidocr import RapidOCR

    model_root = Path(rapidocr.__file__).parent / "models"
    names = {
        "Det": "PP-OCRv6_det_small.onnx",
        "Rec": "PP-OCRv6_rec_small.onnx",
        "Cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    }
    params = {
        "EngineConfig.onnxruntime.intra_op_num_threads": 1,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        "EngineConfig.onnxruntime.use_cuda": False,
        "Global.log_level": "warning",
    }
    models = {}
    for role, name in names.items():
        path = model_root / name
        if not path.is_file():
            raise ValueError("Local OCR model missing: " + str(path))
        params[role + ".model_path"] = str(path)
        models[role] = {"path": str(path), "sha256": sha(path)}
    engine = RapidOCR(params=params)
    args.out.mkdir(parents=True)
    results = []
    for page in pages:
        if sha(page["image"]) != page["image_sha256"]:
            raise ValueError("Original image changed")
        tick = time.monotonic()
        result = engine(page["image"])
        rows = (
            []
            if result.txts is None
            else [
                {"box": box.tolist(), "text": text, "confidence": float(score)}
                for box, text, score in zip(result.boxes, result.txts, result.scores, strict=True)
            ]
        )
        record = {
            "page_id": page["id"],
            "score": page["score"],
            "image": page["image"],
            "image_sha256": page["image_sha256"],
            "seconds": time.monotonic() - tick,
            "all_page_text": rows,
            "source_text_passed_to_ocr": False,
            "note_attachment_scored": False,
        }
        write(args.out / (page["score"] + ".json"), record)
        results.append({"score": page["score"], "seconds": record["seconds"], "regions": len(rows)})
    write(
        args.out / "receipt.json",
        {
            "pages_sha256": sha(args.pages),
            "source_sha256": sha(Path(__file__)),
            "ocr_version": importlib.metadata.version("rapidocr"),
            "models": models,
            "parameters": params,
            "results": results,
            "scope": (
                "All full-page text; not detector fusion, adjudicated transcription, "
                "or note attachment accuracy"
            ),
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
