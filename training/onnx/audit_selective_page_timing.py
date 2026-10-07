"""Score frozen page exports at strict native locations and explicit full-rest flags."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from training.onnx.audit_page_timing_context import edits, event_locations, meter_streams
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import staff_events
from training.onnx.score_paired_page_diagnostic import read_reference


def rest_flags(root, start=0, end=None):
    result, offset = defaultdict(list), 0
    for part in root.findall("part"):
        divisions, staff_count = Fraction(1), 1
        measures = part.findall("measure")
        stop = len(measures) if end is None else end
        if not 0 <= start < stop <= len(measures):
            raise ValueError("Invalid native window")
        for mi, measure in enumerate(measures[:stop]):
            cursor, previous = Fraction(), Fraction()
            for child in measure:
                if child.tag == "attributes":
                    divisions = Fraction(child.findtext("divisions", str(divisions)))
                    staff_count = max(staff_count, int(child.findtext("staves", "1")))
                elif child.tag in ("backup", "forward"):
                    duration = Fraction(child.findtext("duration", "0")) / divisions
                    cursor += duration if child.tag == "forward" else -duration
                elif child.tag == "note":
                    staff = int(child.findtext("staff", "1"))
                    staff_count = max(staff_count, staff)
                    duration = Fraction(child.findtext("duration", "0")) / divisions
                    chord = child.find("chord") is not None
                    onset = previous if chord else cursor
                    if not chord:
                        previous = onset
                        cursor += duration
                    rest = child.find("rest")
                    if mi >= start and rest is not None and child.get("print-object") != "no":
                        result[(offset + staff, mi - start, str(onset), str(duration))].append(
                            rest.get("measure") == "yes"
                        )
        offset += staff_count
    return dict(result)


def compare_flags(reference, control, selective):
    support = matched = gains = regressions = before = after = positives = 0
    for key, labels in reference.items():
        support += len(labels)
        positives += sum(labels)
        if len(labels) != 1 or len(control.get(key, [])) != 1 or len(selective.get(key, [])) != 1:
            continue
        matched += 1
        a, b = control[key][0] == labels[0], selective[key][0] == labels[0]
        before += a
        after += b
        gains += b and not a
        regressions += a and not b
    return dict(
        reference_support=support,
        reference_positives=positives,
        uniquely_matched=matched,
        unmatched=support - matched,
        control_correct=before,
        selective_correct=after,
        improvements=gains,
        regressions=regressions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    page = json.loads((args.case / "observed/identity.json").read_text())["page"]
    proposals = json.loads((args.case / "selective/prediction-only-proposals.json").read_text())
    if not page["boundary_certified"]:
        raise ValueError("Uncertified reference window")
    roots = {}
    for arm in ("control", "selective"):
        path = args.case / "selective" / (arm + ".musicxml")
        if sha(path) != proposals["xml_sha256"][arm]:
            raise ValueError("Frozen XML changed")
        roots[arm] = ET.parse(path).getroot()
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    reference = read_reference(page["reference"])
    start, end = page["start_measure"], page["end_measure"]
    expected = staff_events(reference, start, end)
    meters = meter_streams(reference, start, end)
    report = dict(
        full_rest_flags=compare_flags(
            rest_flags(reference, start, end),
            rest_flags(roots["control"]),
            rest_flags(roots["selective"]),
        ),
        strict_metrics={
            a: dict(
                measure_onset_event=edits(
                    [event_locations(s, True) for s in expected],
                    [event_locations(s, True) for s in staff_events(r)],
                ),
                measure_onset_rhythm=edits(
                    [event_locations(s) for s in expected],
                    [event_locations(s) for s in staff_events(r)],
                ),
                meter_by_measure=edits(meters, meter_streams(r)),
            )
            for a, r in roots.items()
        },
        reference_sha256=page["reference_sha256"],
        source_sha256=sha(__file__),
        limitation="Native staff/measure/onset/duration matching; duplicate and unmatched rests remain in denominator. Explicit source rest flags only. Previously inspected diagnostic pages, not independent qualification.",
    )
    write(args.case / "selective/timing-context-and-flags.json", report)


if __name__ == "__main__":
    main()
