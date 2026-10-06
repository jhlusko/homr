"""Replay scan-adjudicated accidental repairs into an isolated local overlay."""

import argparse
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from contextlib import redirect_stdout
from pathlib import Path

from training.omr_datasets.convert_ossq import _write_example
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.onnx.build_ossq_pixel_review import measure_signature
from training.onnx.prepare_page_certification_extension import preview_window
from training.transformer.training_vocabulary import read_tokens, to_decoder_branches


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timed_fields(text):
    return [
        token.split()[:3]
        for line in text.splitlines()
        for token in line.split("&")
        if token.split()[0].startswith(("note_", "rest_"))
    ]


def merge_lifts(public, before, after):
    """Keep published articulation/slur/order fields; refuse changed music anchors."""
    old, new = timed_fields(before), timed_fields(after)
    if timed_fields(public) != old:
        raise ValueError("Published timed tokens disagree with archived baseline")
    if len(old) != len(new) or any(a[:2] != b[:2] for a, b in zip(old, new)):
        raise ValueError("Repair changes rhythm, pitch position or event count")
    cursor = 0
    rows = []
    for line in public.splitlines():
        tokens = []
        for token in line.split("&"):
            fields = token.split()
            if fields[0].startswith(("note_", "rest_")):
                fields[2] = new[cursor][2]
                cursor += 1
            tokens.append(" ".join(fields))
        rows.append("&".join(tokens))
    return "\n".join(rows), sum(a != b for a, b in zip(old, new))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--public-valid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    adjudication = json.loads(
        (args.recovery / "source-accidental-adjudication-receipt.json").read_text()
    )
    diagnostics = {
        case["system"]: case
        for case in json.loads(
            (args.recovery / "source-accidental-discrepancies.json").read_text()
        )["cases"]
    }
    contexts = {
        c["id"]: c
        for c in json.loads((args.review / "manifest.json").read_text())["systems_with_context"]
    }
    candidates = {
        Path(c["label"]).stem: c
        for c in json.loads((args.recovery / "crop-mapping-candidates.json").read_text())["crops"]
    }
    borodin = {
        c["scan_system"]: c
        for c in json.loads((args.recovery / "borodin-candidate-system-mapping.json").read_text())[
            "pairs"
        ]
    }
    records = []
    xml_changes = 0
    with (args.out / "replay.log").open("w") as log, redirect_stdout(log):
        for decision in adjudication["cases"]:
            source = Path(decision["source"])
            if sha(source) != decision["source_sha256"]:
                raise ValueError("Adjudicated source changed")
            if sha(Path(decision["cleaned_source"])) != decision["cleaned_source_sha256"]:
                raise ValueError("Adjudicated cleaned source changed")
            for path, digest in decision["scan_context_sha256"].items():
                if sha(Path(path)) != digest:
                    raise ValueError("Adjudicated scan context changed")
            context = contexts[decision["system"]]
            key = decision["system"].replace("_", ":")
            if key.startswith("sq10675759"):
                segment = (
                    source.parent
                    / "musicxml/unaligned"
                    / (borodin[key]["symbolic_segment"] + ".musicxml")
                )
            else:
                segment = source.parent / "musicxml/scanned/systemwise" / (key + ".musicxml")
            original = ET.parse(segment).getroot()
            corrected = copy.deepcopy(original)
            root = ET.parse(source).getroot()
            start = decision["candidate_physical_start_index"]
            length = len(original.find("part").findall("measure"))
            canonical = preview_window(root, start, start + length)
            altered = []
            for part_index, (raw, patch, truth) in enumerate(
                zip(original.findall("part"), corrected.findall("part"), canonical.findall("part"))
            ):
                for m_index, (a, b, c) in enumerate(
                    zip(raw.findall("measure"), patch.findall("measure"), truth.findall("measure"))
                ):
                    signatures = [measure_signature(m) for m in (a, c)]

                    def drop_alter(notes):
                        return tuple(note[:3] + note[4:] for note in notes)

                    if drop_alter(signatures[0]) != drop_alter(signatures[1]):
                        raise ValueError("Adjudicated source window changes non-accidental music")
                    old_notes = [n for n in b.findall("note") if n.get("print-object") != "no"]
                    true_notes = [n for n in c.findall("note") if n.get("print-object") != "no"]
                    for n_index, (note, expected) in enumerate(zip(old_notes, true_notes)):
                        old = float(note.findtext("pitch/alter", "0"))
                        new = float(expected.findtext("pitch/alter", "0"))
                        if old != new:
                            if (
                                part_index + 1 not in diagnostics[decision["system"]][
                                    "affected_parts"
                                ]
                            ):
                                raise ValueError("Unreviewed part alteration")
                            if note.find("chord") is not None or (
                                n_index + 1 < len(old_notes)
                                and old_notes[n_index + 1].find("chord") is not None
                            ):
                                raise ValueError(
                                    "Chord alteration requires explicit sidecar reorder audit"
                                )
                            pitch = note.find("pitch")
                            alter = pitch.find("alter")
                            if alter is None:
                                alter = ET.SubElement(pitch, "alter")
                            alter.text = str(int(new))
                            altered.append(
                                dict(
                                    part=part_index + 1,
                                    measure=m_index,
                                    note=n_index,
                                    before=old,
                                    after=new,
                                )
                            )
                if key.startswith("sq10414906"):
                    attrs = patch.find("measure/attributes")
                    if attrs is None:
                        raise ValueError("Missing segment attributes")
                    for child in list(attrs.findall("key")):
                        attrs.remove(child)
                    keys = truth.findall("measure/attributes/key")
                    if len(keys) != 1 or keys[0].findtext("fifths") != "-1":
                        raise ValueError("Adjudicated source key is not one flat")
                    attrs.append(copy.deepcopy(keys[0]))
            xml_changes += len(altered)
            old_xml, new_xml = args.out / (context["id"] + "-old.musicxml"), args.out / (
                context["id"] + "-corrected.musicxml"
            )
            ET.ElementTree(original).write(old_xml)
            ET.ElementTree(corrected).write(new_xml)
            baseline, patched = args.out / "baseline", args.out / "patched"
            baseline.mkdir(exist_ok=True)
            patched.mkdir(exist_ok=True)
            for part in range(4):
                stem = context["id"] + f"_{part + 1}"
                a, _, _ = _write_example(old_xml, part, baseline, stem)
                b, _, _ = _write_example(new_xml, part, patched, stem)
                public = (
                    args.recovery / "borodin-public/scanned/train" / (stem + ".txt")
                    if key.startswith("sq10675759")
                    else args.public_valid / candidates[stem]["label"]
                )
                text, changes = merge_lifts(public.read_text(), a.read_text(), b.read_text())
                key_inserted = False
                if key.startswith("sq10414906"):
                    lines = text.splitlines()
                    if any(line.startswith("keySignature_") for line in lines):
                        raise ValueError("Unexpected published key token")
                    lines.insert(1, "keySignature_-1 . . . . .")
                    text = "\n".join(lines)
                    key_inserted = True
                destination = args.out / (stem + ".txt")
                destination.write_text(text)
                old_sidecar = Path(str(public) + ".notation.json")
                new_sidecar = Path(str(destination) + ".notation.json")
                shutil.copyfile(old_sidecar, new_sidecar)
                symbols = read_tokens(str(destination))
                attached = attach_sidecar(destination, symbols)
                to_decoder_branches(symbols)
                records.append(
                    dict(
                        stem=stem,
                        public=str(public),
                        public_sha256=sha(public),
                        corrected_sha256=sha(destination),
                        lift_changes=changes,
                        key_inserted=key_inserted,
                        annotated_symbols=attached,
                        notation_sidecar_unchanged=sha(old_sidecar) == sha(new_sidecar),
                    )
                )
    assert xml_changes == adjudication["disputed_notes"] == 6
    assert sum(r["lift_changes"] for r in records) == 6
    report = dict(
        crops=len(records),
        xml_note_alterations=xml_changes,
        token_lift_changes=6,
        key_insertions=sum(r["key_inserted"] for r in records),
        annotated_symbols=sum(r["annotated_symbols"] for r in records),
        all_notation_sidecars_unchanged=all(r["notation_sidecar_unchanged"] for r in records),
        public_data_modified=False,
        records=records,
    )
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}))


if __name__ == "__main__":
    main()
