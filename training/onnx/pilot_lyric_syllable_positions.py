"""Replay frozen full-page OCR with character positions enabled, one CPU thread."""

import argparse
import importlib.metadata
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from training.onnx.lyric_syllable_positions import split_positioned_text
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    from rapidocr import RapidOCR

    baseline_receipt = json.loads((args.baseline / "receipt.json").read_text())
    params = dict(baseline_receipt["parameters"])
    models = baseline_receipt["models"]
    for role, model in models.items():
        if sha(model["path"]) != model["sha256"]:
            raise ValueError("OCR graph changed")
        params[role + ".model_path"] = model["path"]
    engine = RapidOCR(params=params)
    for name in ["pilot_lyric_syllable_positions.py", "lyric_syllable_positions.py"]:
        shutil.copyfile(Path(__file__).with_name(name), args.out / name)
    results = []
    for page in json.loads(args.pages.read_text()):
        if page["partition"] != "calibration" or sha(page["image"]) != page["image_sha256"]:
            raise ValueError("Calibration scan changed")
        start = time.monotonic()
        write(args.out / "progress.json", dict(active=page["score"], completed=results))
        cached = json.loads((args.baseline / (page["score"] + ".json")).read_text())
        output = engine(page["image"], return_word_box=True, return_single_char_box=True)
        rows = []
        if output.txts is None or len(output.txts) != len(cached["all_page_text"]):
            raise ValueError("OCR region count changed")
        if len(output.word_results) != len(output.txts):
            raise ValueError("Character output has lost row ownership")
        for index, (box, text, score, character_rows, old) in enumerate(
            zip(
                output.boxes,
                output.txts,
                output.scores,
                output.word_results,
                cached["all_page_text"],
                strict=True,
            )
        ):
            same = text == old["text"] and np.allclose(box, old["box"], rtol=0, atol=0.01)
            characters = [
                dict(index=i, text=c, box=np.asarray(b).tolist())
                for i, (c, confidence, b) in enumerate(character_rows)
            ]
            row = dict(
                box_index=index,
                box=box.tolist(),
                text=text,
                confidence=float(score),
                cached_text_and_box_unchanged=bool(same),
                characters=characters,
            )
            try:
                if not same:
                    raise ValueError("Cached text or parent box changed")
                row["syllables"] = split_positioned_text(text, characters)
            except ValueError as error:
                row["syllables"] = []
                row["geometry_skip"] = str(error)
            rows.append(row)
        write(args.out / (page["score"] + ".json"), dict(page=page, rows=rows))
        results.append(
            dict(
                score=page["score"],
                seconds=time.monotonic() - start,
                regions=len(rows),
                cached_rows_unchanged=sum(r["cached_text_and_box_unchanged"] for r in rows),
                positioned_rows=sum("geometry_skip" not in r for r in rows),
                split_rows=sum(len(r["syllables"]) > 1 for r in rows),
            )
        )
    write(args.out / "progress.json", dict(active=None, completed=results))
    write(
        args.out / "receipt.json",
        dict(
            pages_sha256=sha(args.pages),
            baseline_receipt_sha256=sha(args.baseline / "receipt.json"),
            models=models,
            parameters=params,
            rapidocr_version=importlib.metadata.version("rapidocr"),
            opencv_threads=cv2.getNumThreads(),
            return_word_box=True,
            return_single_char_box=True,
            source_text_passed_to_ocr=False,
            results=results,
            scope="All page text, approximate CTC positions; lyric/verse selection and attachment remain unscored",
        ),
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
