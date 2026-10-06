"""Bind canonical timing to the bounded, adjudicated OSSQ correction overlay."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path

from training.architecture.transformer.exact_timing_targets import HEADS, timing_values
from training.omr_datasets.exact_timing_labels import extract_exact_timing
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.onnx.build_ossq_pixel_review import resolve_symbolic_window
from training.onnx.prepare_page_certification_extension import preview_window
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import read_tokens


def semantic_key(payload):
    if payload is None:
        return None
    fields = {
        k: v
        for k, v in payload.items()
        if k not in ("source_divisions", "source_duration_ticks", "tuplet_containers")
    }
    tuplets = []
    for xml in fields["tuplet_notations_xml"]:
        element = ET.fromstring(xml)
        # Archived round trips dropped these display-only attributes. Restore
        # the complete canonical payload after matching the musical semantics.
        for key in ("bracket", "show-number"):
            element.attrib.pop(key, None)
        tuplets.append(ET.tostring(element, encoding="unicode"))
    fields["tuplet_notations_xml"] = tuplets
    return json.dumps(fields, sort_keys=True)


def part_timings(part):
    divisions = None
    result = []
    for measure in part.findall("measure"):
        divisions = measure.findtext("attributes/divisions", divisions)
        for note in measure.findall("note"):
            if note.get("print-object") != "no":
                result.append(extract_exact_timing(note, divisions))
    return result


def timing_map(source, segment):
    if list(map(semantic_key, source)) != list(map(semantic_key, segment)):
        raise ValueError("Canonical and segment timing differ beyond grid rescaling")
    mapping = {}
    for timing in source:
        key = semantic_key(timing)
        if key in mapping and mapping[key] != timing:
            raise ValueError("Ambiguous canonical source grid for timing class")
        mapping[key] = timing
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("overlay", "adjudication", "target-manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.target_manifest.read_text())
    if manifest["durationPolicy"] != "tick-quantized":
        raise ValueError("Requires frozen tick-quantized classes")
    receipt = json.loads((args.overlay / "receipt.json").read_text())
    windows = json.loads((args.overlay / "source-window-receipt.json").read_text())
    decisions = {c["system"]: c for c in json.loads(args.adjudication.read_text())["cases"]}
    maps = {}
    provenance = []
    for window in windows["windows"]:
        system = window["system"]
        decision = decisions[system]
        source = Path(decision["source"])
        if sha(source) != decision["source_sha256"]:
            raise ValueError("Adjudicated source hash changed")
        root = ET.parse(source).getroot()
        segment_path = args.overlay / (system + "-corrected.musicxml")
        segment = ET.parse(segment_path).getroot()
        bounds = resolve_symbolic_window(root, segment)
        expected = (window["physical_start_index"], window["physical_end_index"])
        if tuple(bounds) != expected:
            raise ValueError("Unique source window changed")
        canonical = preview_window(root, *expected)
        for index, (truth, crop) in enumerate(
            zip(canonical.findall("part"), segment.findall("part"), strict=True), 1
        ):
            maps[system + "_" + str(index)] = timing_map(part_timings(truth), part_timings(crop))
        provenance.append(
            dict(
                window=window,
                source=str(source),
                source_sha256=sha(source),
                corrected_xml_sha256=sha(segment_path),
            )
        )
    args.out.mkdir(parents=True, exist_ok=False)
    groups = {
        split: dict(
            crops=0,
            timed_events=0,
            missing_timing=0,
            heads={h: Counter() for h in HEADS},
            unseen={h: Counter() for h in HEADS},
            reasons=Counter(),
        )
        for split in ("train", "valid")
    }
    records = []
    for record in receipt["records"]:
        stem = record["stem"]
        label = args.overlay / (stem + ".txt")
        public = Path(record["public"])
        if sha(label) != record["corrected_sha256"] or sha(public) != record["public_sha256"]:
            raise ValueError("Correction/public hash changed")
        split = public.parent.name
        group = groups[split]
        group["crops"] += 1
        target = args.out / label.name
        shutil.copyfile(label, target)
        notation = Path(str(label) + ".notation.json")
        shutil.copyfile(notation, Path(str(target) + ".notation.json"))
        payload = json.loads((args.overlay / "patched" / (stem + ".txt.timing.json")).read_text())
        patched_label = args.overlay / "patched" / (stem + ".txt")
        if payload["tokenSha256"] != sha(patched_label):
            raise ValueError("Patched parser sidecar hash changed")
        payload["tokenSha256"] = sha(target)
        for row in payload["symbols"]:
            row["timing"] = maps[stem][semantic_key(row["timing"])]
            timing = row["timing"]
            if timing is None:
                group["missing_timing"] += 1
                continue
            group["timed_events"] += 1
            values, reasons = timing_values(
                timing,
                row["identity"][0].startswith("rest"),
                tick_quarters=Fraction(1) / Fraction(timing["source_divisions"]),
            )
            group["reasons"].update(reasons)
            for head, value in values.items():
                destination = "heads" if value in manifest["classes"][head] else "unseen"
                group[destination][head][value] += 1
        sidecar = Path(str(target) + ".timing.json")
        write(sidecar, payload)
        symbols = read_tokens(target)
        attach_sidecar(target, symbols)  # validates every serialized identity and source grid
        records.append(
            dict(
                stem=stem,
                split=split,
                token_sha256=sha(target),
                timing_sha256=sha(sidecar),
                notation_sha256=sha(notation),
            )
        )
    report = dict(
        groups=groups,
        records=records,
        source_windows=provenance,
        target_manifest=str(args.target_manifest),
        target_manifest_sha256=sha(args.target_manifest),
        overlay_receipt_sha256=sha(args.overlay / "receipt.json"),
        window_receipt_sha256=sha(args.overlay / "source-window-receipt.json"),
        adjudication_sha256=sha(args.adjudication),
        code_sha256=sha(Path(__file__)),
        limitation="12 repaired crops only; classes frozen from training, no validation expansion. "
        "Tokens and notation unchanged from local corrected overlay; published data unchanged.",
    )
    write(args.out / "report.json", report)
    print(json.dumps(groups))


if __name__ == "__main__":
    main()
