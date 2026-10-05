"""Trace source onset gaps versus legacy minimum-duration cursor advances.

Diagnostic only: source onset gaps are never injected into runtime export.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

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


def trace_progression(symbols, events):
    rows = []
    groups = generator.add_tuplet_start_stop(generator.group_into_chords(symbols))
    for group in groups:
        members = [symbol for symbol in group.symbols if hasattr(symbol, "_onset_audit_id")]
        ordinary = [s for s in members if not events[s._onset_audit_id]["grace"]]
        if not ordinary:
            continue
        locations = {
            (
                events[s._onset_audit_id]["part"],
                events[s._onset_audit_id]["physical_measure"],
                events[s._onset_audit_id]["onset_quarters"],
            )
            for s in ordinary
        }
        if len(locations) != 1:
            raise ValueError("Mixed source onsets in diagnostic group")
        part, measure, onset = locations.pop()
        rows.append(
            {
                "part": part,
                "physical_measure": measure,
                "source_onset_quarters": onset,
                "legacy_advance_quarters": str(group.get_render_duration() * 4),
                "members": [{"event": s._onset_audit_id, "token": str(s)} for s in members],
            }
        )
    for row, following in zip(rows, rows[1:]):
        if (row["part"], row["physical_measure"]) != (
            following["part"],
            following["physical_measure"],
        ):
            continue
        gap = Fraction(following["source_onset_quarters"]) - Fraction(row["source_onset_quarters"])
        row["source_next_onset_gap_quarters"] = str(gap)
        row["advance_mismatch"] = gap != Fraction(row["legacy_advance_quarters"])
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, required=True)
    cli.add_argument("--token-name", required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError("Refusing overwrite")
    replay = json.loads(args.replay.read_text())
    matches = [
        (score, crop)
        for score in replay["scores"]
        for crop in score["crops"]
        if Path(crop["token"]).name == args.token_name
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one crop")
    score, crop = matches[0]
    reference = Path(score["reference"])
    if sha(reference) != score["reference_sha256"]:
        raise ValueError("Reference hash mismatch")
    root = read_mxl(reference)
    events = source_events(root)
    voices = traced_parse(root)
    symbols = serialized_copy(
        clean_crop(voices, crop["start_measure"], crop["end_measure"], crop["voice_index"])
    )
    write(
        args.out,
        {
            "reference": str(reference),
            "reference_sha256": sha(reference),
            "crop": crop,
            "progression": trace_progression(symbols, events),
            "export": export_audit(symbols, events, crop["start_measure"]),
            "replay_sha256": sha(args.replay),
            "tool_sha256": sha(Path(__file__)),
            "limitation": (
                "Legacy tokens only; diagnostic source gaps are not runtime targets or predictions."
            ),
        },
    )


if __name__ == "__main__":
    main()
