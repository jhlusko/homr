"""Evaluation-only decomposition of frozen lyric output errors; never rematch notes."""

import argparse
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.native_lyric_attachment_scoring import event_key
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_expanded_lyric_attachments import note_events


def normalized(text):
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold() if c.isalnum())


def decompose(references, predictions, exported, offset, mapping):
    """Partition target multiplicities; native verse permutation is evaluation-only.

    Candidate text is counted only at identical event and mapped verse. Missing
    output is not a detector miss. Decoded identity coverage says nothing about
    independent pixel ownership. Within duplicate keys, allocation is bookkeeping.
    """
    native_ids = defaultdict(set)
    for ref in references:
        native_ids[event_key(ref)].add(ref.get("source_note_ordinal", 0))
    ambiguous = {key for key, ids in native_ids.items() if len(ids) > 1}
    candidates = defaultdict(Counter)
    for pred in predictions:
        key = (event_key(pred["note"], offset), mapping.get(str(pred["physical_row_order"])))
        candidates[key][pred["text"]] += 1
    decoded_keys = {event_key(event, offset) for event in exported}
    records, pending = [], []
    # Consume all exact matches before allocating remaining candidate capacity.
    for ref in references:
        key = event_key(ref)
        record = {"reference": ref}
        texts = candidates[(key, ref["verse"])]
        if key in ambiguous:
            record["category"] = "ambiguous_native_identity"
        elif texts[ref["text"]] > 0:
            record["category"] = "exact"
            texts[ref["text"]] -= 1
        else:
            pending.append(record)
        records.append(record)
    for record in pending:
        ref = record["reference"]
        key = event_key(ref)
        texts = candidates[(key, ref["verse"])]
        available = sorted(text for text, count in texts.items() if count > 0)
        if available:
            text = next(
                (t for t in available if normalized(t) == normalized(ref["text"])), available[0]
            )
            record.update(
                category="text_mismatch_at_exact_note_and_oracle_verse",
                candidate_text=text,
                normalized_text_equal=normalized(text) == normalized(ref["text"]),
                nfc_trimmed_text_equal=(
                    unicodedata.normalize("NFC", text).strip()
                    == unicodedata.normalize("NFC", ref["text"]).strip()
                ),
            )
            texts[text] -= 1
        else:
            record["category"] = (
                "no_attachment_at_exact_decoded_identity"
                if key in decoded_keys
                else "native_identity_absent_from_decoded_voice"
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report_path = args.score / "report.json"
    scored = json.loads(report_path.read_text())
    for name, expected in scored["input_sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen scoring input changed: {name}")
    pages = json.loads(args.pages.read_text())
    root = Path(__file__).resolve().parents[3]
    hashes = {str(report_path): sha(report_path), str(args.pages): sha(args.pages)}
    results = []
    for index, page in enumerate(pages):
        source = root / page["source"]["source"]
        if sha(source) != page["source"]["source_sha256"]:
            raise ValueError("Native source changed")
        parts = [part for part in read_mxl(source).findall("part") if part.findall(".//lyric")]
        if len(parts) != 1:
            raise ValueError("Expected one native lyric-bearing part")
        refs = [
            {**event, "verse": lyric["verse"], "text": lyric["text"]}
            for event in note_events(parts[0])
            if page["start_measure"] < event["physical_measure"] <= page["end_measure"]
            for lyric in event["lyrics"]
        ]
        prediction_path = args.score / f"{page['score']}-predictions.json"
        prediction = json.loads(prediction_path.read_text())
        xml_path = args.decode / f"page-{index}" / "page.musicxml"
        exported = note_events(ET.parse(xml_path).getroot().findall("part")[0])
        metrics = scored["pages"][index]["metrics"]
        if scored["pages"][index]["score"] != page["score"]:
            raise ValueError("Page order differs")
        records = decompose(
            refs,
            prediction["predictions"],
            exported,
            page["start_measure"],
            metrics["oracle_page_verse_mapping"],
        )
        counts = Counter(row["category"] for row in records)
        if counts["exact"] != metrics["joint_literal_matches_after_oracle_page_verse_permutation"]:
            raise ValueError("Audit exact count differs from frozen scorer")
        result = {
            "score": page["score"],
            "counts": dict(counts),
            "normalization_only_text_mismatches": sum(
                r.get("normalized_text_equal", False) for r in records
            ),
            "nfc_trim_only_text_mismatches": sum(
                r.get("nfc_trimmed_text_equal", False) for r in records
            ),
            "targets": records,
            "unresolved_candidate_reasons": prediction["unresolved_reasons"],
        }
        write(args.out / f"{page['score']}.json", result)
        results.append({k: v for k, v in result.items() if k != "targets"})
        hashes.update({str(path): sha(path) for path in [source, prediction_path, xml_path]})
    totals = Counter()
    for result in results:
        totals.update(result["counts"])
    write(
        args.out / "report.json",
        {
            "pages": results,
            "totals": dict(totals),
            "input_sha256": hashes,
            "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
            "scope": (
                "Evaluation-only oracle verse grouping; no note rematching, pixel ownership or detector-miss claims"
            ),
            "production_qualified": False,
        },
    )
    print(json.dumps({"totals": dict(totals), "pages": results}, indent=2))


if __name__ == "__main__":
    main()
