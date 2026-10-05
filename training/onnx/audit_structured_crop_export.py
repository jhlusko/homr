"""Audit source-oracle ties, phrase spans and articulation anchors on real crops.

Oracle arms diagnose label/export capability, never model prediction accuracy.
"""

import argparse
import copy
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch

from homr import music_xml_generator as generator
from homr.transformer.structured_notation import SlurEvent
from homr.transformer.vocabulary import serialized_chords
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.onnx.audit_crop_onsets import source_events, traced_parse
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_lieder_rest_stretch import clean_crop
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import NOTE_ANCHOR_POLICY, read_tokens


def note_inventory(root, *, source):
    """Independent raw XML inventories; numbers are excluded only from endpoint counts."""
    result = {}
    for part_no, part in enumerate(root.findall("part"), 1):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                event = note.get("data-onset-audit-id" if source else "data-original-source-id")
                if not event:
                    continue
                if event in result:
                    raise ValueError("Duplicate diagnostic event identity")
                articulations = []
                for notation in note.findall("notations"):
                    for child in notation:
                        if child.tag in {"slur", "tied", "tuplet", "technical"}:
                            continue
                        if child.tag in {"articulations", "ornaments"}:
                            articulations.extend(f"{child.tag}/{item.tag}" for item in child)
                        else:
                            articulations.append(child.tag)
                result[event] = {
                    "phrase": sorted(s.get("type", "") for s in note.findall("./notations/slur")),
                    "tied": sorted(s.get("type", "") for s in note.findall("./notations/tied")),
                    "sounding_tie": sorted(s.get("type", "") for s in note.findall("tie")),
                    "articulations": sorted(articulations),
                    "rest": note.find("rest") is not None,
                    "source_part": part_no,
                    "source_voice": note.findtext("voice", "1"),
                }
    return result


def phrase_spans(root, *, source, source_scope="voice"):
    """Pair source spans within source voices; emitted XML numbers identify output spans.

    Stops precede starts on one note, matching the documented source slot allocator.
    Unmatched/duplicate endpoints remain explicit findings rather than repaired truth.
    """
    if source_scope not in {"voice", "part"}:
        raise ValueError("Unknown source numbering scope")
    opened = defaultdict(list)
    spans, findings = [], []
    for part_no, part in enumerate(root.findall("part"), 1):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                event = note.get("data-onset-audit-id" if source else "data-original-source-id")
                if not event:
                    continue
                slurs = sorted(
                    note.findall("./notations/slur"), key=lambda s: s.get("type") != "stop"
                )
                for slur in slurs:
                    key = (
                        part_no,
                        (
                            note.findtext("voice", "1")
                            if source and source_scope == "voice"
                            else "xml"
                        ),
                        slur.get("number", "1"),
                    )
                    kind = slur.get("type")
                    if kind == "start":
                        if opened[key]:
                            findings.append({"kind": "duplicate_start", "event": event})
                        opened[key].append(event)
                    elif kind == "stop":
                        if opened[key]:
                            spans.append((opened[key].pop(), event))
                        else:
                            findings.append({"kind": "unmatched_stop", "event": event})
                    elif kind != "continue":
                        findings.append({"kind": "unsupported_type", "event": event, "type": kind})
    for starts in opened.values():
        findings.extend({"kind": "unclosed_start", "event": event} for event in starts)
    return spans, findings


def non_span_hash(root):
    root = copy.deepcopy(root)
    for note in root.findall("./part/measure/note"):
        for tie in note.findall("tie"):
            note.remove(tie)
        for notation in note.findall("notations"):
            for child in list(notation):
                if child.tag in {"slur", "tied"}:
                    notation.remove(child)
    return hashlib.sha256(ET.tostring(root)).hexdigest()


def endpoint_differences(expected, actual, selected):
    differences = defaultdict(list)
    for event in sorted(selected):
        for field in ("phrase", "tied", "sounding_tie", "articulations"):
            if Counter(expected[event][field]) != Counter(actual[event][field]):
                differences[field].append(
                    {
                        "event": event,
                        "source": expected[event][field],
                        "exported": actual[event][field],
                    }
                )
    return dict(differences)


def span_differences(expected_spans, actual_spans, selected):
    interior = Counter(span for span in expected_spans if set(span) <= selected)
    actual = Counter(actual_spans)
    boundary = [span for span in expected_spans if len(set(span) & selected) == 1]
    return {
        "expected_interior_spans": sum(interior.values()),
        "matched_interior_spans": sum((interior & actual).values()),
        "missing_interior_spans": list((interior - actual).elements()),
        "extra_output_pairs": list((actual - interior).elements()),
        "source_boundary_spans": boundary,
    }


def render(symbols, *, arm):
    symbols = copy.deepcopy(symbols)
    for symbol in symbols:
        symbol.exact_timing = None
        if arm == "legacy":
            symbol.notation = None
        elif (
            arm in {"oracle-owned", "oracle-voice-owned", "oracle-identity-strict"}
            and symbol.notation is not None
        ):
            symbol.slur = "_"
    original = generator.build_note_or_rest
    original_slurs = generator.build_slurs

    def build(model_note, *positional, **keywords):
        note = original(model_note, *positional, **keywords)
        if hasattr(model_note, "_onset_audit_id"):
            note.set("data-original-source-id", model_note._onset_audit_id)
        return note

    def voice_owned_slurs(note, symbol, state):
        if arm not in {"oracle-voice-owned", "oracle-identity-strict"} or symbol.notation is None:
            return original_slurs(note, symbol, state)
        notation = note.find("notations")
        if notation is None:
            notation = ET.SubElement(note, "notations")
        staff = generator.get_staff(symbol)
        for slot, (event, side) in enumerate(symbol.notation.slurs, 1):
            # Oracle identity only; model voice classes are within-staff identities.
            key = (symbol._oracle_source_voice, slot)
            placement = {} if str(side) == "unspecified" else {"placement": str(side)}
            if event in {SlurEvent.STOP, SlurEvent.START_AND_STOP}:
                if arm == "oracle-identity-strict" and not state.slur_numbers.get(key):
                    # An unmatched/boundary stop must not close another voice's span.
                    number = state.open_slur(staff=staff)
                    state._release(number)
                else:
                    number = state.close_slur(key, staff)
                ET.SubElement(
                    notation, "slur", {"number": str(number), "type": "stop", **placement}
                )
            if event in {SlurEvent.START, SlurEvent.START_AND_STOP}:
                number = state.open_slur(key, staff)
                ET.SubElement(
                    notation, "slur", {"number": str(number), "type": "start", **placement}
                )

    with (
        patch.object(generator, "build_note_or_rest", build),
        patch.object(generator, "build_slurs", voice_owned_slurs),
        patch.object(
            generator,
            "structured_export_enabled",
            side_effect=lambda head: arm != "legacy" and head in {"tie", "slur"},
        ),
    ):
        return generator.generate_xml(generator.XmlGeneratorArguments(), [symbols], "Diagnostic")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--bundles", type=Path, required=True)
    cli.add_argument("--replay", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing overwrite")
    bundle_report = json.loads(args.bundles.read_text())
    replay = json.loads(args.replay.read_text())
    crop_map = {c["token"]: (s, c) for s in replay["scores"] for c in s["crops"]}
    grouped = defaultdict(list)
    for row in bundle_report["crops"]:
        grouped[row["score"]].append(row)
    counts, results = Counter(), []
    start = time.monotonic()
    for score, rows in grouped.items():
        score_row = crop_map[rows[0]["original_token"]][0]
        path = Path(score_row["reference"])
        if sha(path) != score_row["reference_sha256"]:
            raise ValueError("Source hash mismatch")
        root = read_mxl(path)
        source_events(root)
        expected = note_inventory(root, source=True)
        expected_spans, source_findings = phrase_spans(root, source=True)
        part_spans, part_findings = phrase_spans(root, source=True, source_scope="part")
        questionable = {f["event"] for f in source_findings + part_findings}
        agreed = [
            span
            for span in (Counter(expected_spans) & Counter(part_spans)).elements()
            if not set(span) & questionable
        ]
        ambiguous = list((Counter(expected_spans) - Counter(agreed)).elements())
        voices = traced_parse(root)
        for row in rows:
            crop = crop_map[row["original_token"]][1]
            parsed = clean_crop(
                voices, crop["start_measure"], crop["end_measure"], crop["voice_index"]
            )
            restored = read_tokens(row["token"], expected_annotation_policy=NOTE_ANCHOR_POLICY)
            attach_sidecar(row["token"], restored)
            before = [s for chord in serialized_chords(parsed) for s in chord]
            after = [s for chord in serialized_chords(restored) for s in chord]
            for a, b in zip(before, after, strict=True):
                if (a.rhythm, a.pitch, a.lift, a.position, a.articulation, a.slur, a.notation) != (
                    b.rhythm,
                    b.pitch,
                    b.lift,
                    b.position,
                    b.articulation,
                    b.slur,
                    b.notation,
                ):
                    raise ValueError("Bundle/source alignment mismatch")
                if hasattr(a, "_onset_audit_id"):
                    b._onset_audit_id = a._onset_audit_id
                    source_note = expected[a._onset_audit_id]
                    b._oracle_source_voice = (
                        source_note["source_part"],
                        source_note["source_voice"],
                    )
            selected = {s._onset_audit_id for s in restored if hasattr(s, "_onset_audit_id")}
            arms, controls = {}, {}
            for arm in (
                "legacy",
                "structured-current",
                "oracle-owned",
                "oracle-voice-owned",
                "oracle-identity-strict",
            ):
                xml = render(restored, arm=arm)
                actual = note_inventory(xml, source=False)
                if set(actual) != selected:
                    raise ValueError("Export changed event inventory")
                spans, findings = phrase_spans(xml, source=False)
                diffs = endpoint_differences(expected, actual, selected)
                comparisons = span_differences(expected_spans, spans, selected)
                robust = span_differences(agreed, spans, selected)
                arms[arm] = {
                    "endpoint_differences": diffs,
                    "spans": comparisons,
                    "output_findings": findings,
                    "agreed_unambiguous_spans": robust,
                }
                controls[arm] = non_span_hash(xml)
                for field, changes in diffs.items():
                    counts[f"{arm}.{field}.mismatched_notes"] += len(changes)
                counts[f"{arm}.interior_spans"] += comparisons["expected_interior_spans"]
                counts[f"{arm}.missing_interior_spans"] += len(
                    comparisons["missing_interior_spans"]
                )
                counts[f"{arm}.agreed_interior_spans"] += robust["expected_interior_spans"]
                counts[f"{arm}.missing_agreed_interior_spans"] += len(
                    robust["missing_interior_spans"]
                )
            if len(set(controls.values())) != 1:
                raise ValueError("Oracle arms changed non-span XML, including notes/timing")
            if sha(Path(row["image"])) != row["image_sha256"]:
                raise ValueError("Original image hash mismatch")
            counts["crops"] += 1
            counts["source_events"] += len(selected)
            results.append(
                {
                    "score": score,
                    "token": row["token"],
                    "source": str(path),
                    "source_sha256": sha(path),
                    "crop_start": crop["start_measure"],
                    "arms": arms,
                    "non_span_xml_identical": True,
                    "source_findings": [f for f in source_findings if f["event"] in selected],
                    "source_part_scope_findings": [
                        f for f in part_findings if f["event"] in selected
                    ],
                    "ambiguous_source_spans": [span for span in ambiguous if set(span) <= selected],
                    "impossible_source_ties_on_rests": [
                        e for e in sorted(selected) if expected[e]["rest"] and expected[e]["tied"]
                    ],
                }
            )
        print(score, len(rows), flush=True)
    write(
        args.out / "report.json",
        {
            "crops": results,
            "counts": dict(counts),
            "seconds": time.monotonic() - start,
            "bundle_report_sha256": sha(args.bundles),
            "replay_sha256": sha(args.replay),
            "tool_sha256": sha(Path(__file__)),
            "limitation": (
                "Source-oracle representation/export diagnostics, not predictions. "
                "Raw articulation tags measure ownership, not glyph attributes. "
                "Voice-scoped spans describe the extractor contract, not certified truth. "
                "The agreed view excludes source scope conflicts and findings. "
                "Boundary-only output pairs may be ambiguous; oracle voice is not a model target."
            ),
        },
    )


if __name__ == "__main__":
    main()
