"""Diagnostic source-text coverage score without scan-row or note attachment labels."""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def normalize(text):
    text = text.casefold().replace("’", "'").replace("‘", "'")
    return re.sub(r"[^\wÀ-ÖØ-öø-ÿ']", "", text, flags=re.UNICODE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    pages = json.loads(args.pages.read_text())
    source = [json.loads(line) for line in args.source.read_text().splitlines()]
    selected = {x["page"]["score"]: x for x in json.loads(args.selection.read_text())}
    result = []
    for page in pages:
        item = selected[page["score"]]
        expected = "".join(normalize(row["text"]) for row in source
                             if row["score"] == page["score"]
                             and item["start_measure"] < row["physical_measure"] <= item["end_measure"])
        record_path = args.ocr / (page["score"] + ".json")
        record = json.loads(record_path.read_text())
        if record["image_sha256"] != page["image_sha256"]:
            raise ValueError("OCR/page identity differs")
        actual = "".join(normalize(row["text"]) for row in record.get("all_page_text", []))
        matcher = SequenceMatcher(None, expected, actual, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        result.append({"score": page["score"], "source_entries": sum(row["score"] == page["score"]
                       and item["start_measure"] < row["physical_measure"] <= item["end_measure"]
                       for row in source),
                       "source_characters": len(expected), "ocr_regions": len(record.get("all_page_text", [])),
                       "ocr_characters": len(actual), "ordered_matching_characters": matched,
                       "source_coverage_percent": 100 * matched / len(expected) if expected else None,
                       "sequence_similarity_percent": 100 * matcher.ratio(),
                       "scope": "Ordered character diagnostic; titles/credits and source/scan row order are not separated"})
    write(args.out, {"pages_sha256": sha(args.pages), "source_sha256": sha(args.source),
                     "selection_sha256": sha(args.selection),
                     "ocr_receipt_sha256": sha(args.ocr / "receipt.json"), "results": result,
                     "source_sha256_code": sha(Path(__file__)),
                     "scope": "Native source lyric text versus full-page OCR only; not region precision, "
                              "verse assignment, note attachment or release qualification"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
