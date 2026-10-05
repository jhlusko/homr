"""Diagnostic carryover of explicit predicted meter between certified adjacent pages."""

import argparse
import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write


def terminal_meter(rows):
    """An incomplete later declaration invalidates earlier state."""
    result, pending = None, None
    for row in rows:
        rhythm = row["rhythm"]
        if rhythm.startswith("timeSignatureBeats_"):
            pending = rhythm.split("_", 1)[1]
            result = None
        elif rhythm.startswith("timeSignature/"):
            denominator = rhythm.split("/", 1)[1]
            result = (
                (pending, denominator)
                if pending
                and pending.isdigit()
                and denominator.isdigit()
                and int(pending) > 0
                and int(denominator) > 0
                else None
            )
            pending = None
        else:
            pending = None
    return result


def carry_meter(previous, current):
    if len(previous) != len(current):
        raise ValueError("Physical staff count differs")
    output, receipt = copy.deepcopy(current), []
    for before, after in zip(previous, output, strict=True):
        meter = terminal_meter(before)
        first = next(
            (i for i, s in enumerate(after) if s["rhythm"].startswith(("note_", "rest_"))),
            len(after),
        )
        existing = any(s["rhythm"].startswith("timeSignature") for s in after[:first])
        inserted = meter is not None and not existing
        if inserted:
            template = {
                "pitch": "_",
                "lift": "_",
                "articulation": "_",
                "slur": "_",
                "position": "upper",
            }
            after[first:first] = [
                dict(template, rhythm="timeSignatureBeats_" + meter[0]),
                dict(template, rhythm="timeSignature/" + meter[1]),
            ]
        receipt.append(
            {
                "predicted_meter": meter,
                "existing_initial_declaration": existing,
                "inserted": inserted,
            }
        )
    return output, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "report.json").exists():
        raise ValueError("Refusing to overwrite completed diagnostic")
    setup()
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.slur_crossing import repair_crossings
    from homr.tie_repair import repair_ties
    from homr.tuplet_repair import repair_symbols
    from training.onnx.audit_page_timing_context import edits, event_locations, meter_streams
    from training.onnx.score_full_page_musicxml import compare, staff_events
    from training.onnx.score_paired_page_diagnostic import read_reference
    from training.transformer.review_cached_joint_decoding import restore

    prior = json.loads((args.previous / "comparison.json").read_text())
    report = json.loads((args.current / "comparison.json").read_text())
    page, preceding = report["page"], prior["page"]
    if (
        page["score"] != preceding["score"]
        or page["reference_sha256"] != preceding["reference_sha256"]
        or preceding["end_measure"] != page["start_measure"]
    ):
        raise ValueError("Pages lack independently certified adjacent source windows")
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    reference_root = read_reference(page["reference"])
    reference = staff_events(reference_root, page["start_measure"], page["end_measure"])
    reference_meters = meter_streams(reference_root, page["start_measure"], page["end_measure"])
    results = {}
    for arm in ("accepted", "candidate"):
        identity = json.loads((args.current / arm / "identity.json").read_text())
        previous_identity = json.loads((args.previous / arm / "identity.json").read_text())
        if (
            identity["models"] != previous_identity["models"]
            or identity["policy"] != previous_identity["policy"]
        ):
            raise ValueError("Model/policy differs between pages")
        root_path = Path(__file__).resolve().parents[2]
        for name, digest in identity["code_sha256"].items():
            if sha(root_path / name) != digest:
                raise ValueError("Runtime changed since cached decode: " + name)
        os.environ.update(identity["policy"])
        if (
            identity["policy"]["HOMR_STEM_ARBITRATION"] != "0"
            or identity["policy"]["HOMR_SLUR_SIDE"] != "0"
        ):
            raise ValueError("Unsupported replay policy")
        rows = json.loads((args.current / arm / "decoded-staves.json").read_text())
        previous_rows = json.loads((args.previous / arm / "decoded-staves.json").read_text())
        carried, receipt = carry_meter(previous_rows, rows)
        original = ET.parse(args.current / arm / "page.musicxml").getroot()
        previous_xml = ET.parse(args.previous / arm / "page.musicxml").getroot()

        def topology(root):
            return [
                max(
                    [int(a.text) for a in part.findall(".//attributes/staves")]
                    + [int(n.findtext("staff", "1")) for n in part.findall(".//note")]
                    + [1]
                )
                for part in root.findall("part")
            ]

        if topology(original) != topology(previous_xml):
            raise ValueError("Physical staff grouping differs between pages")
        if sha(args.current / arm / "page.musicxml") != report["arms"][arm]["xml_sha256"]:
            raise ValueError("Current XML changed")
        if sha(args.previous / arm / "page.musicxml") != prior["arms"][arm]["xml_sha256"]:
            raise ValueError("Previous XML changed")
        variants = {}
        for name in ("baseline", "carry", "no_rejection", "carry_no_rejection"):
            voices = [restore(v) for v in (carried if name.startswith("carry") else rows)]
            for i, voice in enumerate(voices):
                repaired, _ = repair_symbols(voice)
                voices[i] = repaired
                repair_beams(repaired)
                repair_ties(repaired)
                repair_crossings(repaired)
            saved = generator.stated_numerator_contradicts_bars
            try:
                if "no_rejection" in name:
                    generator.stated_numerator_contradicts_bars = lambda voice, modal: False
                root = generator.generate_xml(generator.XmlGeneratorArguments(), voices, "")
            finally:
                generator.stated_numerator_contradicts_bars = saved
            events = staff_events(root)
            metrics = compare(reference, events)
            if name == "baseline":
                if (
                    ET.tostring(root) != ET.tostring(original)
                    or metrics != report["arms"][arm]["metrics"]
                ):
                    raise ValueError("Baseline replay differs")
            folder = args.out / arm / name
            folder.mkdir(parents=True, exist_ok=True)
            ET.ElementTree(root).write(
                folder / "page.musicxml", encoding="unicode", xml_declaration=True
            )
            variants[name] = {
                "metrics": metrics,
                "strict_events": edits(
                    [event_locations(v, True) for v in reference],
                    [event_locations(v, True) for v in events],
                ),
                "meter_errors": edits(reference_meters, meter_streams(root)),
                "exact_original_xml": ET.tostring(root) == ET.tostring(original),
                "predicted_meter_streams": meter_streams(root),
                "xml_sha256": sha(folder / "page.musicxml"),
            }
        results[arm] = {
            "carry_receipt": receipt,
            "variants": variants,
            "previous_cached_sha256": sha(args.previous / arm / "decoded-staves.json"),
            "current_cached_sha256": sha(args.current / arm / "decoded-staves.json"),
        }
    write(
        args.out / "report.json",
        {
            "arms": results,
            "previous_page": preceding,
            "current_page": page,
            "diagnostic_only": True,
            "reference_used_only_for_scoring": True,
            "source_sha256": sha(Path(__file__)),
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
