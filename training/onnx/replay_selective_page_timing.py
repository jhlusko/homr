"""Research selective page export; references enter only after both XML arms freeze."""

import argparse
import copy
import json
import os
import pickle
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path

from homr import music_xml_generator as generator
from homr.tuplet_repair import is_single_staff
from training.onnx.probe_selective_timing_export import bars, meter_values, note_invariants, propose
from training.onnx.qualify_free_timing_probe import legacy_timing, serialize
from training.onnx.qualify_joint_export import sha, write


def independent_meter(symbols, excluded):
    if not is_single_staff(symbols):
        return None, "mixed-staff"
    denominators = [
        s.rhythm.split("/")[1] for s in symbols if s.rhythm.startswith("timeSignature/")
    ]
    numerators = [
        int(s.rhythm.split("_")[-1]) for s in symbols if s.rhythm.startswith("timeSignatureBeats_")
    ]
    if len(denominators) != 1 or len(numerators) > 1:
        return None, "missing-or-changing-meter"
    lengths = [
        sum((g.get_duration() * 4 for g in bar), Fraction())
        for i, bar in enumerate(bars(symbols))
        if i != excluded
        and any(s.rhythm.startswith("note") and "G" not in s.rhythm for g in bar for s in g.symbols)
    ]
    ranked = Counter(lengths).most_common()
    if len(lengths) < 3 or not ranked or ranked[0][1] * 4 < len(lengths) * 3:
        return None, "weak-independent-bar-context"
    meter = ranked[0][0]
    beats = meter * int(denominators[0]) / 4
    if beats.denominator != 1 or (numerators and beats != numerators[0]):
        return None, "meter-declaration-conflict"
    return meter, "independent-other-bars"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.observation / "identity.json").read_text())
    receipt = json.loads((args.observation / "probe-receipt.json").read_text())
    if not receipt["accepted_xml_exact_match"]:
        raise ValueError("Require verified accepted control")
    os.environ.update(identity["policy"])
    # This pickle is produced locally by our image-only observer, never downloaded.
    with (args.observation / "final-export-input.pkl").open("rb") as handle:
        arguments, voices, title = pickle.load(handle)
    control = generator.generate_xml(arguments, copy.deepcopy(voices), title)
    if ET.tostring(control, encoding="unicode") != ET.tostring(
        ET.parse(args.observation / "page.musicxml").getroot(), encoding="unicode"
    ):
        raise ValueError("Captured v7 input does not reproduce accepted control")
    candidate = copy.deepcopy(voices)
    changes, contexts = [], []
    for vi, voice in enumerate(voices):
        predictions = []
        for i, symbol in enumerate(voice):
            symbol.probe_index = i
            candidate[vi][i].probe_index = i
            value = getattr(symbol, "probe_timing", None)
            predictions.append(
                value
                or {
                    "timing.ratio": "1:1",
                    "timing.elapsed": "0",
                    "timing.written": "unknown",
                    **legacy_timing(serialize(symbol)),
                    "timing.full_rest": "no",
                }
            )
        for bi, _ in enumerate(bars(voice)):
            meter, reason = independent_meter(voice, bi)
            proposed, edits, refusals = propose(voice, predictions, meter)
            chosen = [
                e
                for e in edits
                if e["crop_measure"] == bi + 1
                and getattr(voice[e["index"]], "probe_timing", None) is not None
            ]
            for edit in chosen:
                candidate[vi][edit["index"]] = proposed[edit["index"]]
                changes.append(dict(edit, voice=vi + 1))
            contexts.append(
                dict(
                    voice=vi + 1, measure=bi + 1, meter=str(meter), reason=reason, refusals=refusals
                )
            )
    original = generator.build_note_or_rest

    def flag(symbol, *pos):
        note = original(symbol, *pos)
        if getattr(symbol, "probe_full_rest", False):
            note.find("rest").set("measure", "yes")
        return note

    try:
        generator.build_note_or_rest = flag
        selective = generator.generate_xml(arguments, candidate, title)
    finally:
        generator.build_note_or_rest = original
    if (
        note_invariants(control) != note_invariants(selective)
        or meter_values(control) != meter_values(selective)
        or len(control.findall(".//measure")) != len(selective.findall(".//measure"))
    ):
        raise ValueError("Candidate export changed protected recognition, notation or meter")
    for arm, root in [("control", control), ("selective", selective)]:
        ET.ElementTree(root).write(
            args.out / (arm + ".musicxml"), encoding="unicode", xml_declaration=True
        )
    write(
        args.out / "prediction-only-proposals.json",
        dict(
            changes=changes,
            contexts=contexts,
            reference_opened=False,
            input_sha256=sha(args.observation / "final-export-input.pkl"),
            source_sha256=sha(__file__),
            xml_sha256={a: sha(args.out / (a + ".musicxml")) for a in ("control", "selective")},
        ),
    )
    # All proposals and XML bytes are frozen before reference scoring.
    from training.onnx.score_full_page_musicxml import compare, staff_events
    from training.onnx.score_paired_page_diagnostic import read_reference

    page = identity["page"]
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    reference = staff_events(
        read_reference(page["reference"]), page["start_measure"], page["end_measure"]
    )
    write(
        args.out / "report.json",
        dict(
            changes=changes,
            context_reasons=Counter(c["reason"] for c in contexts),
            metrics={
                a: compare(reference, staff_events(r))
                for a, r in [("control", control), ("selective", selective)]
            },
            limitation="Previously evaluated page; policy designed after crop results. No independent qualification or promotion.",
        ),
    )


if __name__ == "__main__":
    main()
