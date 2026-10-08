"""Explain conservative timing decisions using frozen decoded export inputs only."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def timeline_failures(notes, staff, span):
    reasons = []
    if any(n["onset"] < 0 or n["onset"] + n["duration"] > span for n in notes):
        reasons.append("part-overflow-or-negative-onset")
    lanes = defaultdict(dict)
    for note in notes:
        lane = lanes[(note["staff"], note["voice"])]
        lane[note["onset"]] = max(lane.get(note["onset"], Fraction()), note["duration"])
    for key, lane in lanes.items():
        end = Fraction()
        for onset, duration in sorted(lane.items()):
            if onset < end:
                reasons.append(f"overlap-in-staff-voice-{key}")
            end = onset + duration
    target = [n for n in notes if n["staff"] == staff]
    if not target:
        reasons.append("no-target-notes")
    elif len({n["voice"] for n in target}) != 1:
        reasons.append("multiple-target-voice-identifiers")
    elif (
        min(n["onset"] for n in target) != 0
        or max(n["onset"] + n["duration"] for n in target) != span
    ):
        reasons.append("target-does-not-cover-complete-span")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root))
    from homr.music_xml_generator import _generate_xml
    from homr.tremolo_timing import (
        apply,
        bars,
        context_meter,
        lane_symbols,
        peer_span,
        render,
        safe_bar,
    )

    # This must be an agent-created local snapshot, never an untrusted download.
    with args.input.open("rb") as handle:
        arguments, voices, title = pickle.load(handle)  # noqa: S301 - agent-created local snapshot
    _, _, baseline = render(_generate_xml, arguments, voices, title)
    edits, contexts = [], []
    for pi, voice in enumerate(voices):
        for bi, groups in enumerate(bars(voice)):
            if not any("tremolo" in s.articulation for group in groups for s in group.symbols):
                continue
            span, reason = context_meter(voices, pi, bi, allow_repeated=True)
            if span is None:
                span, reason = peer_span(voices, pi, bi, baseline, safe_bar)
            row = {
                "part": pi + 1,
                "measure": bi + 1,
                "span": str(span),
                "context": reason,
                "selectedCounts": {},
            }
            for position, staff in (("upper", 1), ("lower", 2)):
                selected = lane_symbols(groups, position, span) if span else []
                ids = {id(s) for s in selected}
                indices = [i for i, symbol in enumerate(voice) if id(symbol) in ids]
                row["selectedCounts"][position] = len(indices)
                if indices:
                    edits.append(
                        {
                            "part": pi,
                            "bar": bi,
                            "staff": staff,
                            "indices": indices,
                            "meter": str(span),
                        }
                    )
            contexts.append(row)
    marks = {n["key"]: n["tuplet_mark"] for n in baseline if n["key"] is not None}
    changed = {(e["part"], i) for e in edits for i in e["indices"]}
    _, _, timeline = render(
        _generate_xml, arguments, apply(voices, edits), title, marks=marks, changed=changed
    )
    decisions = []
    for edit in edits:
        notes = [
            n
            for n in timeline
            if n["part"] == edit["part"] and n["bar"] == edit["bar"] and not n["grace"]
        ]
        span = Fraction(edit["meter"])
        decisions.append(
            {
                "proposal": edit,
                "safe": safe_bar(timeline, edit["part"], edit["bar"], edit["staff"], span),
                "reasons": timeline_failures(notes, edit["staff"], span),
                "timeline": notes,
            }
        )
    report = {
        "runtimeRoot": str(args.runtime_root),
        "input": str(args.input),
        "inputSha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "structuredExportFamilies": os.environ.get("HOMR_STRUCTURED_EXPORT"),
        "pairedTimingArgument": arguments.paired_tremolo_timing,
        "contexts": contexts,
        "decisions": decisions,
        "scope": "Actual frozen symbol-state diagnostic; no source reference used",
    }
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                "contexts": contexts,
                "decisions": [{k: v for k, v in d.items() if k != "timeline"} for d in decisions],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
