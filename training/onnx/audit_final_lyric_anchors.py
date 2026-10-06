"""Check diagnostic provenance survives decoding without changing saved XML."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_page_notation_policy import anchors
from training.onnx.link_lyric_anchor_sequences import exact_sequence_matches, source_notes
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    reports = []
    for index, page in enumerate(json.loads(args.pages.read_text())):
        folder = args.pilot / f"page-{index}"
        baseline = args.baseline / f"page-{index}"
        xml = ET.parse(folder / "page.musicxml").getroot()
        xml_equal = ET.tostring(xml) == ET.tostring(ET.parse(baseline / "page.musicxml").getroot())
        cache_equal = json.loads((folder / "decoded-staves.json").read_text()) == json.loads(
            (baseline / "decoded-staves.json").read_text()
        )
        if not xml_equal or not cache_equal:
            raise ValueError("Instrumentation changed runtime outputs")
        final = json.loads((folder / "final-note-anchors.json").read_text())
        vocal = [r for r in final if r["voice_index"] == 0]
        exported_vocal = source_notes(xml.findall("part")[0])
        source = workspace / page["source"]["source"]
        if sha(source) != page["source"]["source_sha256"]:
            raise ValueError("Native source changed")
        native = read_mxl(source)
        ref = {
            k: v
            for k, v in anchors(native, page["start_measure"], page["end_measure"]).items()
            if k[0] == 1
        }
        pred = {k: v for k, v in anchors(xml).items() if k[0] == 1}
        common = {k for k in ref.keys() & pred.keys() if len(ref[k]) == len(pred[k]) == 1}
        reports.append(
            dict(
                score=page["score"],
                exact_xml=xml_equal,
                exact_cached_symbols=cache_equal,
                final_notes=len(final),
                final_coordinates_retained=sum(r["page_coordinate"] is not None for r in final),
                vocal_notes=len(vocal),
                vocal_coordinates_retained=sum(r["page_coordinate"] is not None for r in vocal),
                vocal_final_decode_matches_xml_pitch_duration=exact_sequence_matches(
                    exported_vocal, vocal
                ),
                native_vocal_pitched_events=sum(map(len, ref.values())),
                unique_native_vocal_xml_timing_pitch_duration_matches=len(common),
                unmatched_native_anchor_keys=len(set(ref) - set(pred)),
                unmatched_predicted_anchor_keys=len(set(pred) - set(ref)),
                note_pixel_ownership_certified=False,
                input_sha256={
                    str(p): sha(p)
                    for p in [
                        folder / "final-note-anchors.json",
                        folder / "page.musicxml",
                        baseline / "page.musicxml",
                    ]
                },
            )
        )
    report = dict(
        auditor_sha256=sha(Path(__file__)),
        pages=reports,
        scope="Two calibration pages; diagnostic first-pass provenance retained through final decode, checked against exported vocal XML; no automatic source linkage/fusion deployment",
    )
    write(args.out, report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
