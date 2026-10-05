"""Conditional scan-row OCR character errors; no GT text alignment or note scoring."""

import argparse
import json
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from training.onnx.qualify_joint_export import sha, write


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("’", "'").replace("‘", "'")
    return "".join(c for c in text if not c.isspace() and c not in "-_—–·")


def row_for_box(box, rows):
    x = sum(p[0] for p in box) / len(box)
    y = sum(p[1] for p in box) / len(box)
    matches = [
        r for r in rows if r["box"][0] <= x <= r["box"][2] and r["box"][1] <= y <= r["box"][3]
    ]
    if len(matches) > 1:
        raise ValueError("OCR center belongs to ambiguous reference rows")
    return matches[0]["id"] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detector-area", type=int, choices=(4, 200))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    receipt = json.loads((args.references.parent / "receipt.json").read_text())
    if receipt["partition"] != "calibration" or receipt["rows_sha256"] != sha(args.references):
        raise ValueError("Reference receipt mismatch")
    refs = json.loads(args.references.read_text())
    results, inputs = [], {}
    for score in sorted({r["score"] for r in refs}):
        path = args.ocr / (score + ".json")
        inputs[score] = sha(path)
        prediction = json.loads(path.read_text())
        rows = [r for r in refs if r["score"] == score]
        if any(r["image_sha256"] != prediction["image_sha256"] for r in rows):
            raise ValueError("Original page differs")
        assigned = {r["id"]: [] for r in rows}
        unassigned = 0
        candidates = prediction.get("all_page_text")
        if args.detector_area is not None:
            candidates = [
                dict(
                    item,
                    box=[
                        [item["left"], item["top"]],
                        [item["right"], item["top"]],
                        [item["right"], item["bottom"]],
                        [item["left"], item["bottom"]],
                    ],
                )
                for item in prediction["areas"][str(args.detector_area)]
            ]
        if candidates is None:
            raise ValueError("Prediction type requires explicit detector area")
        for item in candidates:
            key = row_for_box(item["box"], rows)
            if key is None:
                unassigned += 1
            else:
                assigned[key].append(item)
        for row in rows:
            items = sorted(assigned[row["id"]], key=lambda r: min(p[0] for p in r["box"]))
            predicted = " ".join(r["text"] for r in items)
            expected, actual = normalize(row["visible_text"]), normalize(predicted)
            results.append(
                {
                    "id": row["id"],
                    "score": score,
                    "source_verse": row["source_verse"],
                    "visible_reference": row["visible_text"],
                    "predicted_text": predicted,
                    "errors": Levenshtein.distance(expected, actual),
                    "support": len(expected),
                    "detected_regions": len(items),
                    "unassigned_page_regions": unassigned,
                }
            )
    errors, support = sum(r["errors"] for r in results), sum(r["support"] for r in results)
    report = {
        "detector_area": args.detector_area,
        "rows": results,
        "errors": errors,
        "support": support,
        "character_error_percent": 100 * errors / support if support else None,
        "reference_sha256": sha(args.references),
        "ocr_sha256": inputs,
        "scorer_sha256": sha(Path(__file__)),
        "reference_receipt": receipt,
        "scope": (
            "Calibration-only conditional row transcription. Fixed manual spatial rows, "
            "not lyric-region precision, words/syllable accuracy, "
            "production fusion or note attachment."
        ),
    }
    write(args.out, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
