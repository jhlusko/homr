"""Audit frozen fixed-policy exports across native measures and corpora."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from training.onnx.audit_page_timing_context import edits, event_locations
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import compare, staff_events
from training.onnx.score_paired_page_diagnostic import read_reference


def native_measures(reference, control, selective):
    count = (
        max(
            (
                e["measure"]
                for streams in (reference, control, selective)
                for staff in streams
                for e in staff
            ),
            default=-1,
        )
        + 1
    )
    result = []
    for mi in range(count):
        r, a, b = [
            [[e for e in s if e["measure"] == mi] for s in streams]
            for streams in (reference, control, selective)
        ]
        record = dict(
            page_bar=mi + 1,
            control=compare(r, a),
            selective=compare(r, b),
            control_onset=edits(
                [event_locations(s, True) for s in r], [event_locations(s, True) for s in a]
            ),
            selective_onset=edits(
                [event_locations(s, True) for s in r], [event_locations(s, True) for s in b]
            ),
            control_selective_events_equal=a == b,
        )
        record["regressed"] = [
            m
            for m in ("event", "pitch", "rhythm", "accidental")
            if record["selective"][m]["errors"] > record["control"][m]["errors"]
        ]
        if record["selective_onset"]["errors"] > record["control_onset"]["errors"]:
            record["regressed"].append("strict_onset_event")
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while args.wait and not (args.run / "completion.json").exists():
        time.sleep(10)
    completion = json.loads((args.run / "completion.json").read_text())
    jobs = json.loads((args.run / "jobs.json").read_text())
    cases = []
    totals = defaultdict(lambda: defaultdict(lambda: dict(errors=0, support=0)))
    for job in jobs:
        name = job["name"]
        status = completion["jobs"][name]
        if status["status"] != "ok":
            cases.append(dict(name=name, **status))
            continue
        directory = args.run / name / "tremolo-rule"
        identity = json.loads((Path(job["case"]) / "observed/identity.json").read_text())
        page = identity["page"]
        proposals = json.loads((directory / "prediction-only-proposals.json").read_text())
        if proposals["source_sha256"] != completion["policy_sha256"]:
            raise ValueError("Mixed policy versions")
        if proposals.get("peer_span_sha256") != completion.get("peer_span_sha256"):
            raise ValueError("Mixed peer context versions")
        roots = {}
        for arm in ("control", "selective"):
            path = directory / (arm + ".musicxml")
            if sha(path) != proposals["xml_sha256"][arm]:
                raise ValueError("Frozen output changed")
            roots[arm] = ET.parse(path).getroot()
        if sha(page["reference"]) != page["reference_sha256"]:
            raise ValueError("Reference changed")
        reference = staff_events(
            read_reference(page["reference"]), page["start_measure"], page["end_measure"]
        )
        a, b = [staff_events(roots[arm]) for arm in ("control", "selective")]
        measures = native_measures(reference, a, b)
        affected = {e["bar"] + 1 for e in proposals["edits"]}
        if any(
            not m["control_selective_events_equal"] and m["page_bar"] not in affected
            for m in measures
        ):
            raise ValueError("Untargeted native measure changed")
        write(
            directory / "per-native-measure-audit.json",
            dict(
                measures=measures,
                reference_start=page["start_measure"],
                regressed_measures=[m["page_bar"] for m in measures if m["regressed"]],
                source_sha256=sha(__file__),
            ),
        )
        metrics = {
            arm: compare(reference, events) for arm, events in [("control", a), ("selective", b)]
        }
        onset = {
            arm: edits(
                [event_locations(s, True) for s in reference],
                [event_locations(s, True) for s in events],
            )
            for arm, events in [("control", a), ("selective", b)]
        }
        row = dict(
            name=name,
            status="ok",
            source=page["source"],
            score=page["score"],
            cached_control=job["cached"],
            changed_symbols=sum(len(e["indices"]) for e in proposals["edits"]),
            changed_bars=len(affected),
            metrics=metrics,
            strict_onset=onset,
            regressed_native_measures=[m["page_bar"] for m in measures if m["regressed"]],
            targeted_voice_changes=proposals["targeted_voice_changes"],
        )
        cases.append(row)
        group = page["source"] + ("/cached-controls" if job["cached"] else "/new-scores")
        for arm in ("control", "selective"):
            for metric in ("event", "pitch", "rhythm", "accidental"):
                tally = totals[group][arm + "/" + metric]
                tally["errors"] += metrics[arm][metric]["errors"]
                tally["support"] += metrics[arm][metric]["support"]
            tally = totals[group][arm + "/strict_onset_event"]
            tally["errors"] += onset[arm]["errors"]
            tally["support"] += onset[arm]["support"]
    write(
        args.run / "audit-report.json",
        dict(
            cases=cases,
            totals=totals,
            attempted=len(jobs),
            successful=sum(r["status"] == "ok" for r in cases),
            source_sha256=sha(__file__),
            completion_sha256=sha(args.run / "completion.json"),
            adaptive_replay=(args.run / "selection-receipt.json").exists()
            and "source_run" in json.loads((args.run / "selection-receipt.json").read_text()),
            limitation="Fixed research rule, new scores disjoint from policy development; recognition/model exposure tracked, not untouched test. Ground truth only scores frozen exports. Cached controls are reported separately. Failures remain visible.",
        ),
    )


if __name__ == "__main__":
    main()
