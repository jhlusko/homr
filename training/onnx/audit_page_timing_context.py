"""Add strict measure/onset and inherited-meter diagnostics to completed page pairs.

No reference material enters inference. Relative physical measure indices are
compared directly; no oracle rebar or model-dependent alignment is performed.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import staff_events
from training.onnx.score_paired_page_diagnostic import read_reference


def meter_streams(root, start=0, end=None):
    streams = []
    for part in root.findall("part"):
        measures = part.findall("measure")
        stop = len(measures) if end is None else end
        if not 0 <= start < stop <= len(measures):
            raise ValueError("Invalid meter window")
        staff_count, default, numbered = 1, None, {}
        rows = []
        for index, measure in enumerate(measures[:stop]):
            material_seen = False
            for attributes in measure:
                if attributes.tag in ("note", "forward", "backup"):
                    material_seen = True
                if attributes.tag != "attributes":
                    continue
                if material_seen and attributes.find("time") is not None:
                    raise ValueError("Unsupported meter change after measure material")
                staff_count = max(staff_count, int(attributes.findtext("staves", "1")))
                for time in attributes.findall("time"):
                    beats = tuple(c.text for c in time.findall("beats"))
                    denominators = tuple(c.text for c in time.findall("beat-type"))
                    meter = (beats, denominators, time.find("senza-misura") is not None)
                    if time.get("number"):
                        numbered[int(time.get("number"))] = meter
                    else:
                        default = meter
                        numbered = {}
            staff_count = max(
                staff_count,
                max((int(n.findtext("staff", "1")) for n in measure.findall("note")), default=1),
            )
            if index >= start:
                rows.append((index - start, default, dict(numbered)))
        for staff in range(1, staff_count + 1):
            streams.append(
                [(index, numbered.get(staff, default)) for index, default, numbered in rows]
            )
    return streams


def event_locations(events, include_pitch=False):
    return [
        (
            e["measure"],
            str(Fraction(e["onset"])),
            *((tuple(e["pitch"]) if e["pitch"] is not None else None,) if include_pitch else ()),
            e["duration"],
            e["grace"],
            e["pitch"] is None,
        )
        for e in events
    ]


def edits(reference, predicted):
    errors = support = 0
    for index in range(max(len(reference), len(predicted))):
        a = reference[index] if index < len(reference) else []
        b = predicted[index] if index < len(predicted) else []
        support += len(a)
        errors += Levenshtein.distance(a, b)
    return {
        "errors": errors,
        "support": support,
        "error_percent": 100 * errors / support if support else None,
    }


def audit(directory):
    report_path = directory / "comparison.json"
    report = json.loads(report_path.read_text())
    page = report["page"]
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    root = read_reference(page["reference"])
    a, z = page["start_measure"], page["end_measure"]
    reference = staff_events(root, a, z)
    reference_meters = meter_streams(root, a, z)
    arms = {}
    for name in ("accepted", "candidate"):
        record = report["arms"][name]
        if record.get("status", "ok") != "ok":
            arms[name] = {"status": "failed", "error": record["error"]}
            continue
        xml = directory / name / "page.musicxml"
        if sha(xml) != record["xml_sha256"]:
            raise ValueError("Prediction changed")
        prediction = ET.parse(xml).getroot()
        events = staff_events(prediction)
        meters = meter_streams(prediction)
        metrics = {
            "measure_onset_rhythm": edits(
                [event_locations(s) for s in reference], [event_locations(s) for s in events]
            ),
            "measure_onset_event": edits(
                [event_locations(s, True) for s in reference],
                [event_locations(s, True) for s in events],
            ),
            "meter_by_measure": edits(reference_meters, meters),
        }
        arms[name] = {
            "status": "ok",
            "metrics": metrics,
            "xml_meter_by_staff": meters,
            "reference_measure_counts_by_staff": [len(s) for s in reference_meters],
            "predicted_measure_counts_by_staff": [len(s) for s in meters],
            "predicted_unknown_meter_measures": sum(m is None for s in meters for _, m in s),
            "reference_unknown_meter_measures": sum(
                m is None for s in reference_meters for _, m in s
            ),
        }
    write(
        directory / "timing-context-audit.json",
        {
            "page": page,
            "comparison_sha256": sha(report_path),
            "tool_sha256": sha(__file__),
            "event_reader_sha256": sha(Path(__file__).with_name("score_full_page_musicxml.py")),
            "arms": arms,
            "reference_meter_by_staff": reference_meters,
            "scope": (
                "Strict page-relative physical-measure/onset event edits and "
                "inherited-meter edits. "
                "Missing/extra bars or altered bar placement count as errors; no oracle alignment. "
                "Not crop SER, independent voice-ownership accuracy, notation "
                "accuracy or promotion gate. "
                "XML-declared meters may be exporter-inferred; they are not proof "
                "of model meter emission."
            ),
        },
    )
    return arms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run.resolve()), indent=2))
