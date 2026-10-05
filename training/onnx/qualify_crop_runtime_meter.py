"""Paired old/new runtime guards on the actual meter-affected Lieder crop labels."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from homr import music_xml_generator as generator
from training.onnx.audit_crop_onsets import (
    export_audit,
    serialized_copy,
    source_events,
    traced_parse,
)
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_lieder_rest_stretch import clean_crop
from training.onnx.qualify_joint_export import sha, write
from training.onnx.qualify_meter_guard import load_old_guard, without_times
from training.transformer.training_vocabulary import token_lines_to_str


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, required=True)
    cli.add_argument("--crop-gate", type=Path, required=True)
    cli.add_argument("--old-generator", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError("Refusing overwrite")
    selected = {c["token"] for c in json.loads(args.crop_gate.read_text())["changed_crops"]}
    replay = json.loads(args.replay.read_text())
    old_guard = load_old_guard(args.old_generator)
    new_guard = generator.stated_numerator_contradicts_bars
    generate = generator.generate_xml
    results = []
    for score in replay["scores"]:
        crops = [c for c in score["crops"] if c["token"] in selected]
        if not crops:
            continue
        reference = Path(score["reference"])
        if sha(reference) != score["reference_sha256"]:
            raise ValueError("Source hash mismatch")
        root = read_mxl(reference)
        events = source_events(root)
        voices = traced_parse(root)
        for crop in crops:
            symbols = clean_crop(
                voices, crop["start_measure"], crop["end_measure"], crop["voice_index"]
            )
            if token_lines_to_str(symbols) != Path(crop["replayed_label"]).read_text():
                raise ValueError("Current label replay mismatch")
            arms, xmls = {}, {}
            for name, guard in (("old", old_guard), ("new", new_guard)):

                def capture(*positional, _arm=name, _outputs=xmls, **keywords):
                    xml = generate(*positional, **keywords)
                    _outputs[_arm] = xml
                    return xml

                with (
                    patch.object(generator, "generate_xml", capture),
                    patch.object(generator, "stated_numerator_contradicts_bars", guard),
                ):
                    arms[name] = export_audit(
                        serialized_copy(symbols), events, crop["start_measure"]
                    )
                if arms[name].get("error"):
                    raise ValueError(f"Crop export failed: {crop['token']}: {arms[name]}")
            if without_times(xmls["old"]) != without_times(xmls["new"]):
                raise ValueError(f"Non-time crop XML changed: {crop['token']}")
            if arms["old"] != arms["new"]:
                raise ValueError("Meter guard changed event audit")

            def time_map(xml):
                return {
                    (m, t.get("number", "all")): (t.findtext("beats"), t.findtext("beat-type"))
                    for m, measure in enumerate(xml.findall("./part/measure"), 1)
                    for t in measure.findall("./attributes/time")
                }

            before, after = time_map(xmls["old"]), time_map(xmls["new"])
            changes = []
            for key in sorted(set(before) | set(after)):
                if before.get(key) == after.get(key):
                    continue
                measure = xmls["new"].findall("./part/measure")[key[0] - 1]
                note = next(n for n in measure.findall("note") if n.get("data-original-source-id"))
                source = events[note.get("data-original-source-id")]
                expected = source["time_signature"]
                if after.get(key) != (expected["beats"], expected["beat_type"]):
                    raise ValueError(f"Exported crop meter disagrees with source: {crop['token']}")
                changes.append(
                    {
                        "crop_measure": key[0],
                        "source_part": source["part"],
                        "source_measure": source["physical_measure"],
                        "old": before.get(key),
                        "new": after.get(key),
                        "source": expected,
                    }
                )
            results.append(
                {
                    "token": crop["token"],
                    "changes": changes,
                    "non_time_xml_identical": True,
                    "event_audit_identical": True,
                    "event_audit": arms["new"],
                }
            )
    if len(results) != len(selected):
        raise ValueError("Missing selected crops")
    write(
        args.out,
        {
            "crops": results,
            "count": len(results),
            "changed_crops": sum(bool(r["changes"]) for r in results),
            "changed_declarations": sum(len(r["changes"]) for r in results),
            "replay_sha256": sha(args.replay),
            "crop_gate_sha256": sha(args.crop_gate),
            "old_generator_sha256": sha(args.old_generator),
            "tool_sha256": sha(Path(__file__)),
            "limitation": "Actual current legacy labels; source sidecars absent, no predictions.",
        },
    )


if __name__ == "__main__":
    main()
