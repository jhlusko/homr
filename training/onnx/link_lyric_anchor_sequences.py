"""Propose source ownership only for count-exact vocal system sequences.

This is a calibration linkage diagnostic, not automatic certification: written
pitch/duration equality does not certify pixels, accidentals, or final XML ownership.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write


def source_notes(part):
    divisions = Fraction(1)
    result = []
    for measure_index, measure in enumerate(part.findall("measure"), 1):
        cursor = previous = Fraction(0)
        ordinal = 0
        for element in measure:
            if element.tag == "attributes":
                divisions = Fraction(element.findtext("divisions", str(divisions)))
            elif element.tag in ("backup", "forward"):
                change = Fraction(element.findtext("duration", "0")) / divisions
                cursor += change if element.tag == "forward" else -change
            elif element.tag == "note":
                ordinal += 1
                duration = Fraction(element.findtext("duration", "0")) / divisions
                chord = element.find("chord") is not None
                onset = previous if chord else cursor
                if not chord:
                    previous = onset
                    cursor += duration
                pitch = element.find("pitch")
                if pitch is None:
                    continue
                result.append(
                    dict(
                        source_part=part.get("id"),
                        physical_measure=measure_index,
                        source_note_ordinal=ordinal,
                        onset_quarters=str(onset),
                        duration_quarters=str(duration),
                        pitch=pitch.findtext("step") + pitch.findtext("octave"),
                        lyrics=[
                            dict(
                                verse=x.get("number", "1"),
                                text="".join(t.text or "" for t in x.findall("text")),
                            )
                            for x in element.findall("lyric")
                        ],
                    )
                )
    return result


def exact_sequence_matches(reference, prediction):
    return [(r["pitch"], Fraction(r["duration_quarters"])) for r in reference] == [
        (p["pitch"], EncodedSymbol(p["rhythm"]).get_duration().fraction * 4) for p in prediction
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    workspace = Path(__file__).resolve().parents[3]
    for page_index, page in enumerate(json.loads(args.pages.read_text())):
        source = workspace / page["source"]["source"]
        if sha(source) != page["source"]["source_sha256"]:
            raise ValueError("Native source changed")
        root = read_mxl(source)
        parts = [part for part in root.findall("part") if part.findall(".//lyric")]
        if len(parts) != 1 or root.findall(".//note/unpitched"):
            raise ValueError("Pilot requires one known lyric-bearing source part")
        native = source_notes(parts[0])
        rows = json.loads((args.pilot / f"page-{page_index}" / "note-anchors.json").read_text())
        systems = sorted(page["systems"], key=lambda s: s["scan_index"])
        records = []
        # The runtime decodes voice-major, system-minor. This assumption remains
        # diagnostic and is shown alongside the original scan for verification.
        for staff_call, system in enumerate(systems):
            reference = [
                r
                for r in native
                if system["start_measure"] < r["physical_measure"] <= system["end_measure"]
            ]
            prediction = [r for r in rows if r["staff_call"] == staff_call]
            exact = exact_sequence_matches(reference, prediction)
            records.append(
                dict(
                    scan_system=system["scan_index"],
                    staff_call=staff_call,
                    source_notes=len(reference),
                    predicted_notes=len(prediction),
                    exact_written_pitch_duration_sequence=exact,
                    proposed_links=(
                        [dict(source=r, prediction=p) for r, p in zip(reference, prediction)]
                        if exact
                        else []
                    ),
                    note_pixel_attachment_certified=False,
                )
            )
        reports.append(dict(score=page["score"], systems=records))
    write(
        args.out,
        dict(
            pages=reports,
            source_used_for_inference=False,
            scope="Exact written-pitch/duration linkage proposals only; source-ownership/pixels remain unverified",
        ),
    )
    print(
        json.dumps(
            [
                {
                    "score": p["score"],
                    "systems": [
                        {k: v for k, v in s.items() if k != "proposed_links"} for s in p["systems"]
                    ],
                }
                for p in reports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
