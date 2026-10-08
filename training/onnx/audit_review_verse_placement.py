"""Audit cached source-free placements without oracle verse-number permutation.

Physical row order as native verse number is explicitly a diagnostic assumption;
source verse IDs may reflect edition numbering rather than printed row position.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.native_lyric_attachment_scoring import event_key
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_expanded_lyric_attachments import note_events


def exact_verse_matches(references, predictions, offset):
    refs = Counter((event_key(r), str(r["verse"]), r["text"]) for r in references)
    pred = Counter(
        (event_key(p["note"], offset), str(p["physical_row_order"]), p["text"]) for p in predictions
    )
    return sum((refs & pred).values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads((args.predictions / "report.json").read_text())
    frozen = previous["input_sha256"]
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise ValueError("Original scoring input changed")
    pages = json.loads(args.pages.read_text())
    results = []
    workspace = Path(__file__).resolve().parents[3]
    for page in pages:
        cached = args.predictions / (page["score"] + "-predictions.json")
        result = json.loads(cached.read_text())
        if result["native_input_used"] or not result["export_trace"]["exact_xml"]:
            raise ValueError("Require source-free exact-export-traced cached predictions")
        source = workspace / page["source"]["source"]
        if sha(source) != page["source"]["source_sha256"]:
            raise ValueError("Native evaluation source changed")
        parts = [p for p in read_mxl(source).findall("part") if p.findall(".//lyric")]
        if len(parts) != 1:
            raise ValueError("Need unique lyric source part")
        refs = [
            {**e, "verse": lyric["verse"], "text": lyric["text"]}
            for e in note_events(parts[0])
            if page["start_measure"] < e["physical_measure"] <= page["end_measure"]
            for lyric in e["lyrics"]
        ]
        old = next(r for r in previous["pages"] if r["score"] == page["score"])
        results.append(
            {
                "score": page["score"],
                "reference_targets": len(refs),
                "cached_predictions": len(result["predictions"]),
                "literal_matches_assuming_physical_row_equals_native_verse": exact_verse_matches(
                    refs, result["predictions"], page["start_measure"]
                ),
                "previous_oracle_verse_matches": old["metrics"][
                    "joint_literal_matches_after_oracle_page_verse_permutation"
                ],
                "matches_ignoring_verse": old["metrics"]["joint_literal_matches_ignoring_verse"],
                "previous_oracle_mapping": old["metrics"]["oracle_page_verse_mapping"],
                "predictions_sha256": sha(cached),
            }
        )
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "report.json",
        {
            "pages": results,
            "source_sha256": sha(__file__),
            "previous_report_sha256": sha(args.predictions / "report.json"),
            "scope": (
                "Cached placement numbering diagnostic, not fresh inference or independent "
                "pixel ownership; does not prove wrong printed verse placement"
            ),
            "source_roles_entered_inference": False,
            "reviewed_text_inserted": False,
            "automatic_attachment": False,
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
