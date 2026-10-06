"""Trace native missing-output targets through source-free cached attachment diagnostics."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.onnx.native_lyric_attachment_scoring import event_key
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_expanded_lyric_attachments import note_events


def trace_target(ref, exported, final, prediction, offset, mapping):
    indices = [i for i, event in enumerate(exported) if event_key(event, offset) == event_key(ref)]
    if len(indices) != 1:
        return {"stage": "nonunique_exported_identity", "exported_indices": indices}
    origins = [
        record["origin"]
        for record in prediction["export_trace"]["records"]
        if record["part_index"] == 0 and record["exported_index"] == indices[0]
    ]
    if len(origins) != 1 or origins[0] is None:
        return {"stage": "nonunique_export_origin"}
    final_indices = [
        i for i, row in enumerate(final) if [row["voice_index"], row["token_index"]] == origins[0]
    ]
    if len(final_indices) != 1:
        return {"stage": "nonunique_final_origin"}
    index = final_indices[0]
    row = final[index]
    system = next(s for s in prediction["systems"] if s["staff_call"] == row["staff_call"])
    base = {
        "final_note_index": index,
        "staff_call": row["staff_call"],
        "page_coordinate": row["page_coordinate"],
        "system_index": row["system_index"],
    }
    heads = [int(h) for h, i in system.get("head_to_final_note", {}).items() if i == index]
    if len(heads) != 1:
        return {**base, "stage": "no_accepted_notehead_association"}
    head = heads[0]
    candidates = [
        trace
        for trace in prediction["attachment_trace"]
        if trace.get("staff_call") == row["staff_call"] and trace.get("nearest_head_index") == head
    ]
    verse_candidates = [
        trace
        for trace in candidates
        if mapping.get(str(trace.get("physical_row_order"))) == ref["verse"]
    ]
    return {
        **base,
        "head_index": head,
        "head_x": system["head_xs"][head],
        "stage": (
            "candidate_at_expected_row_requires_capacity_audit"
            if verse_candidates
            else (
                "nearest_head_candidates_only_in_other_physical_rows"
                if candidates
                else "no_routed_syllable_selects_accepted_head"
            )
        ),
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["audit", "trace", "decode", "pages", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    audit = json.loads((args.audit / "report.json").read_text())
    traced = json.loads((args.trace / "report.json").read_text())
    for report in [audit, traced]:
        for name, expected in report["input_sha256"].items():
            if sha(Path(name)) != expected:
                raise ValueError(f"Frozen input changed: {name}")
    pages = json.loads(args.pages.read_text())
    totals, results, hashes = Counter(), [], {}
    for index, page in enumerate(pages):
        audit_path = args.audit / f"{page['score']}.json"
        trace_path = args.trace / f"{page['score']}-predictions.json"
        final_path = args.decode / f"page-{index}" / "final-note-anchors.json"
        xml_path = args.decode / f"page-{index}" / "page.musicxml"
        target_audit = json.loads(audit_path.read_text())
        prediction = json.loads(trace_path.read_text())
        final = [r for r in json.loads(final_path.read_text()) if r["voice_index"] == 0]
        exported = note_events(ET.parse(xml_path).getroot().findall("part")[0])
        mapping = traced["pages"][index]["metrics"]["oracle_page_verse_mapping"]
        records = [
            {
                "reference": target["reference"],
                **trace_target(
                    target["reference"],
                    exported,
                    final,
                    prediction,
                    page["start_measure"],
                    mapping,
                ),
            }
            for target in target_audit["targets"]
            if target["category"] == "no_attachment_at_exact_decoded_identity"
        ]
        counts = Counter(r["stage"] for r in records)
        totals.update(counts)
        write(
            args.out / f"{page['score']}.json",
            {"score": page["score"], "counts": dict(counts), "targets": records},
        )
        results.append({"score": page["score"], "counts": dict(counts)})
        hashes.update({str(p): sha(p) for p in [audit_path, trace_path, final_path, xml_path]})
    for p in [args.pages, args.audit / "report.json", args.trace / "report.json"]:
        hashes[str(p)] = sha(p)
    write(
        args.out / "report.json",
        {
            "totals": dict(totals),
            "pages": results,
            "input_sha256": hashes,
            "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
            "scope": (
                "Evaluation-only native output trace; no independent pixel ownership or detector-miss claims"
            ),
        },
    )
    print(json.dumps({"totals": dict(totals), "pages": results}, indent=2))


if __name__ == "__main__":
    main()
