"""Replay cached physical row recovery and audit interpolated vocal pitch geometry."""

import argparse
import json
from pathlib import Path

from training.onnx.lyric_notehead_association import pitch_staff_position
from training.onnx.lyric_row_routing import route_rows
from training.onnx.lyric_staff_pitch_geometry import interpolate_notehead_positions
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-page", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    original = read(args.two_page / "row-routing-report.json")
    routing = []
    for index, page in enumerate(original["pages"]):
        transforms = read(args.two_page / f"page-{index}/staff-transforms.json")
        after = route_rows(page["rows"], transforms, recover_row_neighbors=True)
        routing.append(
            {
                "score": page["score"],
                "population": "two-page row-recovery selection",
                "before_routed": len(page["routed"]),
                "before_unresolved": len(page["unresolved"]),
                "rows": page["rows"],
                **after,
            }
        )
    completed = read(args.expanded / "completion.json")
    for page in completed["pages"]:
        if page["status"] != "ok":
            raise ValueError("Expanded geometry failure must be resolved or reported separately")
        folder = args.expanded / f"page-{page['page_index']}"
        before = read(folder / "routing.json")
        after = route_rows(
            before["rows"], read(folder / "staff-transforms.json"), recover_row_neighbors=True
        )
        record = {
            "score": page["score"],
            "population": "expanded area200/10% detector selection",
            "before_routed": len(before["routed"]),
            "before_unresolved": len(before["unresolved"]),
            "rows": before["rows"],
            **after,
        }
        write(args.out / f"{page['score']}-routing.json", record)
        routing.append(record)
    pitch = []
    for index, page in enumerate(read(args.linkage)["pages"]):
        transforms = read(args.two_page / f"page-{index}/staff-transforms.json")
        voices = read(args.two_page / f"page-{index}/decoded-staves.json")
        clefs = [r["rhythm"] for r in voices[0] if r["rhythm"].startswith("clef_")]
        if len(clefs) != 1:
            raise ValueError("Only one predicted vocal clef is supported")
        for system in page["systems"]:
            if not system["proposed_links"]:
                continue
            transform = next(t for t in transforms if t["staff_call"] == system["staff_call"])
            heads = interpolate_notehead_positions(transform)
            if not system["exact_written_pitch_duration_sequence"] or len(heads) != len(
                system["proposed_links"]
            ):
                raise ValueError("Reference geometry evaluation is not count exact")
            for head, link in zip(heads, system["proposed_links"]):
                expected = pitch_staff_position(link["prediction"]["pitch"], clefs[0])
                pitch.append(
                    {
                        "score": page["score"],
                        "system": system["scan_system"],
                        **head,
                        "decoded_position": expected,
                        "interpolated_agrees": head["staff_position"] == expected,
                        "stored_agrees": head["stored_staff_position"] == expected,
                        "native_reference": link["source"],
                    }
                )
    summaries = [
        {
            "score": r["score"],
            "population": r["population"],
            "selected": len(r["rows"]),
            "before_routed": r["before_routed"],
            "after_routed": len(r["routed"]),
            "recovered": sum("recovery" in row for row in r["routed"]),
            "remaining_unresolved": len(r["unresolved"]),
        }
        for r in routing
    ]
    report = {
        "scope": (
            "Ten calibration pages with different frozen selection populations; recovery coverage and conditional 70-note vocal pitch audit only"
        ),
        "source_used_by_recovery_or_interpolation": False,
        "production_qualified": False,
        "routing_summary": summaries,
        "two_page_routing": routing[:2],
        "vocal_pitch_audit": pitch,
        "input_sha256": hashes,
        "code_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("lyric_row_routing.py"),
                Path(__file__).with_name("lyric_staff_pitch_geometry.py"),
            ]
        },
    }
    write(args.out / "geometry-report.json", report)
    print(json.dumps(summaries, indent=2))
    print(
        "Pitch agreement stored/interpolated:",
        sum(r["stored_agrees"] for r in pitch),
        sum(r["interpolated_agrees"] for r in pitch),
        "of",
        len(pitch),
    )


if __name__ == "__main__":
    main()
