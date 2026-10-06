"""Diagnostic scan-only attachments, evaluated against native page-window targets."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.link_lyric_anchor_sequences import source_notes
from training.onnx.lyric_notehead_association import (
    associate_x,
    pitch_staff_position,
    reject_pitch_conflicts,
    reject_sequence_ambiguity,
)
from training.onnx.lyric_row_routing import route_rows
from training.onnx.lyric_staff_pitch_geometry import interpolate_notehead_positions
from training.onnx.native_lyric_attachment_scoring import score_native_outputs
from training.onnx.qualify_joint_export import sha, write
from training.onnx.trace_cached_lyric_export import trace_cached_export


def note_events(part):
    events = source_notes(part)
    pitches = [
        n.find("pitch") for n in part.findall("./measure/note") if n.find("pitch") is not None
    ]
    if len(pitches) != len(events):
        raise ValueError("Pitched XML event counts differ")
    for event, pitch in zip(events, pitches):
        event["alter"] = pitch.findtext("alter", "0")
    return events


def predict(folder, selected, positioned, *, trace_export=False):
    """No native inputs. First runtime voice is an explicit diagnostic assumption."""
    transforms = json.loads((folder / "staff-transforms.json").read_text())
    voices = json.loads((folder / "decoded-staves.json").read_text())
    final = [
        r
        for r in json.loads((folder / "final-note-anchors.json").read_text())
        if r["voice_index"] == 0
    ]
    exported = note_events(ET.parse(folder / "page.musicxml").getroot().findall("part")[0])
    # Check the complete pitched-note sequence before linking by index.
    xml_equal = len(final) == len(exported) and all(
        r["pitch"] == e["pitch"]
        and EncodedSymbol(r["rhythm"]).get_duration().fraction * 4
        == Fraction(e["duration_quarters"])
        for r, e in zip(final, exported)
    )
    export_indices = {i: i for i in range(len(final))} if xml_equal else {}
    export_trace = None
    if trace_export:
        export_trace = trace_cached_export(folder)
        origins = {}
        for record in export_trace["records"]:
            if record["part_index"] == 0 and record["origin"] is not None:
                origins.setdefault(tuple(record["origin"]), []).append(record["exported_index"])
        export_indices = {
            i: origins[(0, row["token_index"])][0]
            for i, row in enumerate(final)
            if len(origins.get((0, row["token_index"]), [])) == 1
        }
    rows = selected["all_page_text"]
    routing = route_rows(rows, transforms, recover_row_neighbors=True)
    positions_by_box = {tuple(tuple(p) for p in row["box"]): row for row in positioned["rows"]}
    predictions, unresolved, systems = [], [], []
    clef = None
    clefs_at = {}
    for index, token in enumerate(voices[0]):
        if token["rhythm"].startswith("clef_"):
            clef = token["rhythm"]
        clefs_at[index] = clef
    vocal_calls = {r.get("staff_call") for r in final if r.get("staff_call") is not None}
    association = {}
    for call in sorted(vocal_calls):
        indexed = [(index, row) for index, row in enumerate(final) if row.get("staff_call") == call]
        transform = next(t for t in transforms if t["staff_call"] == call)
        status = {
            "staff_call": call,
            "final_notes": len(indexed),
            "detected_heads": len(transform["detected_noteheads"]),
        }
        try:
            if any(i not in export_indices for i, _ in indexed):
                raise ValueError("No unique final-token to exported-note provenance")
            token_indices = [row["token_index"] for _, row in indexed]
            chord_groups = []
            for i in range(1, len(token_indices)):
                if any(
                    token["rhythm"] == "chord"
                    for token in voices[0][token_indices[i - 1] + 1 : token_indices[i]]
                ):
                    if chord_groups and chord_groups[-1][-1] == i - 1:
                        chord_groups[-1].append(i)
                    else:
                        chord_groups.append([i - 1, i])
            if any(row["page_coordinate"] is None for _, row in indexed):
                raise ValueError("Missing final-note page coordinate")
            heads = interpolate_notehead_positions(transform)
            decoded_positions = [
                pitch_staff_position(row["pitch"], clefs_at[row["token_index"]] or "unknown")
                for _, row in indexed
            ]
            detected_positions = [head["staff_position"] for head in heads]
            head_xs = [
                head["center"][0] * transform["original_to_processed_ratio"][0] for head in heads
            ]
            matched = reject_sequence_ambiguity(
                reject_pitch_conflicts(
                    associate_x(
                        [row["page_coordinate"][0] for _, row in indexed],
                        head_xs,
                        max_distance=40,
                        ambiguity_margin=5,
                    ),
                    decoded_positions,
                    detected_positions,
                ),
                decoded_positions,
                detected_positions,
                detected_x=head_xs,
                chord_groups=chord_groups,
            )
            association[call] = {
                "head_xs": head_xs,
                "head_to_note": {j: indexed[i][0] for i, j in matched["pairs"]},
            }
            status.update(
                status="ok",
                associated=len(matched["pairs"]),
                unresolved_final=len(matched["unresolved_decoded"]),
                unresolved_chord_notes=sum(len(group) for group in chord_groups),
            )
        except (ValueError, TypeError) as error:
            status.update(status="unresolved", reason=str(error))
        systems.append(status)
    row_routes = {r["index"]: r for r in routing["routed"]}
    for index, row in enumerate(rows):
        positioned_row = positions_by_box.get(tuple(tuple(p) for p in row["box"]))
        if (
            positioned_row is None
            or positioned_row["text"] != row["text"]
            or not positioned_row["cached_text_and_box_unchanged"]
        ):
            raise ValueError("Frozen OCR/positioned row identity differs")
        for syllable_index, syllable in enumerate(positioned_row.get("syllables", [])):
            record = {
                "selected_row_index": index,
                "syllable_index": syllable_index,
                "text": syllable["text"],
            }
            routed = row_routes.get(index)
            if routed is None or "physical_row_order" not in routed:
                unresolved.append({**record, "reason": "unresolved_physical_row"})
                continue
            geometry = association.get(routed["staff_call"])
            if geometry is None or not geometry["head_xs"]:
                unresolved.append({**record, "reason": "unsupported_or_nonvocal_staff"})
                continue
            candidates = sorted(
                (abs(syllable["center"][0] - x), j) for j, x in enumerate(geometry["head_xs"])
            )
            if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-9:
                unresolved.append({**record, "reason": "ambiguous_nearest_head"})
                continue
            note_index = geometry["head_to_note"].get(candidates[0][1])
            if note_index is None:
                unresolved.append({**record, "reason": "nearest_head_has_no_accepted_decoded_note"})
                continue
            predictions.append(
                {
                    **record,
                    "physical_row_order": routed["physical_row_order"],
                    "native_verse": None,
                    "note": exported[export_indices[note_index]],
                    "nearest_head_distance_pixels": candidates[0][0],
                    "system_index": final[note_index]["system_index"],
                }
            )
    return {
        "predictions": predictions,
        "unresolved_syllables": unresolved,
        "unresolved_reasons": dict(Counter(r["reason"] for r in unresolved)),
        "systems": systems,
        "final_xml_pitch_duration_sequence_exact": xml_equal,
        "export_trace": export_trace,
        "vocal_voice_assumption": "First runtime voice; not general instrument identification",
        "selection": selected["selection"],
        "native_input_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-export", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    pages = json.loads(args.pages.read_text())
    results, hashes = [], {str(args.pages): sha(args.pages)}
    for index, page in enumerate(pages):
        folder = args.decode / f"page-{index}"
        record = {"score": page["score"], "page_index": index}
        if (
            not (folder / "result.json").exists()
            or json.loads((folder / "result.json").read_text())["status"] != "ok"
        ):
            record.update(status="decode_unavailable")
            results.append(record)
            continue
        identity = json.loads((folder / "identity.json").read_text())
        if identity["page"]["image_sha256"] != page["image_sha256"]:
            raise ValueError("Decode/evaluation images differ")
        selected_path = args.selected / f"{page['score']}.json"
        positions_path = args.positions / f"{page['score']}.json"
        selected, positioned = json.loads(selected_path.read_text()), json.loads(
            positions_path.read_text()
        )
        if (
            selected["image_sha256"] != page["image_sha256"]
            or positioned["page"]["image_sha256"] != page["image_sha256"]
        ):
            raise ValueError("OCR/evaluation images differ")
        prediction = predict(folder, selected, positioned, trace_export=args.trace_export)
        write(args.out / f"{page['score']}-predictions.json", prediction)
        source = root / page["source"]["source"]
        if sha(source) != page["source"]["source_sha256"]:
            raise ValueError("Native source changed")
        native = read_mxl(source)
        lyric_parts = [part for part in native.findall("part") if part.findall(".//lyric")]
        if len(lyric_parts) != 1:
            raise ValueError("Evaluation requires one known lyric-bearing source part")
        references = [
            {**event, "verse": lyric["verse"], "text": lyric["text"]}
            for event in note_events(lyric_parts[0])
            if page["start_measure"] < event["physical_measure"] <= page["end_measure"]
            for lyric in event["lyrics"]
        ]
        metrics = score_native_outputs(references, prediction["predictions"], page["start_measure"])
        record.update(
            status="ok",
            metrics=metrics,
            expected_manifest_targets=page["native_targets"],
            systems=prediction["systems"],
            unresolved_reasons=prediction["unresolved_reasons"],
            final_xml_pitch_duration_sequence_exact=prediction[
                "final_xml_pitch_duration_sequence_exact"
            ],
            exact_export_replay=bool(
                prediction["export_trace"] and prediction["export_trace"]["exact_xml"]
            ),
        )
        results.append(record)
        for path in [
            source,
            selected_path,
            positions_path,
            *(
                folder / name
                for name in (
                    "identity.json",
                    "page.musicxml",
                    "decoded-staves.json",
                    "final-note-anchors.json",
                    "staff-transforms.json",
                )
            ),
        ]:
            hashes[str(path)] = sha(path)
    code = [
        Path(__file__),
        Path(__file__).with_name("native_lyric_attachment_scoring.py"),
        Path(__file__).with_name("lyric_notehead_association.py"),
        Path(__file__).with_name("lyric_staff_pitch_geometry.py"),
        Path(__file__).with_name("lyric_row_routing.py"),
        Path(__file__).with_name("trace_cached_lyric_export.py"),
    ]
    report = {
        "pages": results,
        "production_qualified": False,
        "scope": (
            "Calibration native-output joint literal recovery with oracle page-global verse permutation; not independent box ownership, detector misses or native verse-ID accuracy"
        ),
        "input_sha256": hashes,
        "code_sha256": {str(path): sha(path) for path in code},
    }
    write(args.out / "report.json", report)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
