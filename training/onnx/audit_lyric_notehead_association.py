"""Audit geometry-only matching using count-exact calibration and simulated omissions."""

import argparse
import json
from pathlib import Path

from training.onnx.lyric_notehead_association import (
    associate_x,
    pitch_staff_position,
    reject_pitch_conflicts,
    reject_sequence_ambiguity,
)
from training.onnx.lyric_staff_pitch_geometry import interpolate_notehead_positions
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--geometry", type=Path, help="Optional replay with segmented staff positions"
    )
    parser.add_argument("--pitch-gate", action="store_true")
    parser.add_argument("--sequence-gate", action="store_true")
    parser.add_argument("--interpolated-positions", action="store_true")
    args = parser.parse_args()
    if args.sequence_gate and not args.pitch_gate:
        raise ValueError("Sequence gate requires pitch gate")
    if args.interpolated_positions and not args.pitch_gate:
        raise ValueError("Interpolated positions require pitch gate")
    linkage_path = args.anchors / "source-linkage.json"
    pages = json.loads(linkage_path.read_text())["pages"]
    totals = {}
    cases = []
    errors = []
    hashes = {str(linkage_path): sha(linkage_path)}
    for page_index, page in enumerate(pages):
        geometry = args.geometry or args.anchors
        transform_path = geometry / f"page-{page_index}/staff-transforms.json"
        hashes[str(transform_path)] = sha(transform_path)
        transforms = json.loads(transform_path.read_text())
        if args.geometry:
            original_notes = args.anchors / f"page-{page_index}/note-anchors.json"
            replay_notes = geometry / f"page-{page_index}/note-anchors.json"
            for path in (original_notes, replay_notes):
                hashes[str(path)] = sha(path)
            if json.loads(original_notes.read_text()) != json.loads(replay_notes.read_text()):
                raise ValueError("Replay predictions differ from the frozen calibration linkage")
        if args.pitch_gate:
            decoded_path = geometry / f"page-{page_index}/decoded-staves.json"
            hashes[str(decoded_path)] = sha(decoded_path)
            vocal = json.loads(decoded_path.read_text())[0]
            clefs = [row["rhythm"] for row in vocal if row["rhythm"].startswith("clef_")]
            if len(clefs) != 1:
                raise ValueError("Diagnostic supports a single predicted vocal clef only")
        for system in page["systems"]:
            links = system["proposed_links"]
            if not links:
                continue
            transform = next(t for t in transforms if t["staff_call"] == system["staff_call"])
            heads = sorted(
                h[0] * transform["original_to_processed_ratio"][0]
                for h in transform["detected_noteheads"]
            )
            decoded = [link["prediction"]["page_coordinate"][0] for link in links]
            if args.pitch_gate:
                if any(len(point["y"]) != 5 for point in transform["staff_grid"]):
                    raise ValueError("Grandstaff pitch mapping is not qualified")
                positioned_heads = sorted(
                    transform["detected_notehead_positions"], key=lambda h: h["center"][0]
                )
                if args.interpolated_positions:
                    positioned_heads = interpolate_notehead_positions(transform)
                if [
                    h["center"][0] * transform["original_to_processed_ratio"][0]
                    for h in positioned_heads
                ] != heads:
                    raise ValueError("Segmented position identities differ from head coordinates")
                decoded_positions = [
                    pitch_staff_position(link["prediction"]["pitch"], clefs[0]) for link in links
                ]
                detected_positions = [h["staff_position"] for h in positioned_heads]
            if not system["exact_written_pitch_duration_sequence"] or len(heads) != len(decoded):
                raise ValueError("Calibration reference mapping is not count exact")
            # Sensitivity sweep in original page pixels; no production default selected.
            for distance in (20, 40, 60):
                variants = [("original", list(enumerate(decoded)), list(enumerate(heads)))]
                for removed in range(len(heads)):
                    variants.append(
                        (
                            "drop_head",
                            list(enumerate(decoded)),
                            [(i, x) for i, x in enumerate(heads) if i != removed],
                        )
                    )
                    variants.append(
                        (
                            "drop_decode",
                            [(i, x) for i, x in enumerate(decoded) if i != removed],
                            list(enumerate(heads)),
                        )
                    )
                    variants.append(
                        (
                            "duplicate_head",
                            list(enumerate(decoded)),
                            [*enumerate(heads), (-1, heads[removed])],
                        )
                    )
                for kind, notes, detections in variants:
                    result = associate_x(
                        [x for _, x in notes],
                        [x for _, x in detections],
                        max_distance=distance,
                        ambiguity_margin=5,
                    )
                    if args.pitch_gate:
                        result = reject_pitch_conflicts(
                            result,
                            [decoded_positions[i] for i, _ in notes],
                            [
                                detected_positions[i] if i >= 0 else detected_positions[removed]
                                for i, _ in detections
                            ],
                        )
                        if args.sequence_gate:
                            result = reject_sequence_ambiguity(
                                result,
                                [decoded_positions[i] for i, _ in notes],
                                [
                                    detected_positions[i] if i >= 0 else detected_positions[removed]
                                    for i, _ in detections
                                ],
                            )
                    correct = sum(notes[i][0] == detections[j][0] for i, j in result["pairs"])
                    for i, j in result["pairs"]:
                        if notes[i][0] != detections[j][0]:
                            errors.append(
                                {
                                    "score": page["score"],
                                    "system": system["scan_system"],
                                    "variant": kind,
                                    "distance": distance,
                                    "remaining_decoded_indices": [index for index, _ in notes],
                                    "decoded_index": notes[i][0],
                                    "detected_index": detections[j][0],
                                    "expected_source_note": links[notes[i][0]]["source"],
                                    "assigned_source_note": links[detections[j][0]]["source"],
                                }
                            )
                    available = len({i for i, _ in notes} & {i for i, _ in detections})
                    key = f"{kind}:distance={distance}"
                    total = totals.setdefault(
                        key,
                        {
                            "cases": 0,
                            "available_pairs": 0,
                            "matched": 0,
                            "correct": 0,
                            "wrong": 0,
                            "unresolved_decoded": 0,
                        },
                    )
                    for field, value in {
                        "cases": 1,
                        "available_pairs": available,
                        "matched": len(result["pairs"]),
                        "correct": correct,
                        "wrong": len(result["pairs"]) - correct,
                        "unresolved_decoded": len(result["unresolved_decoded"]),
                    }.items():
                        total[field] += value
                    if kind == "original":
                        cases.append(
                            {
                                "score": page["score"],
                                "system": system["scan_system"],
                                "distance": distance,
                                **result,
                            }
                        )
    report = {
        "scope": (
            "Two selected calibration pages; simulated omissions/duplicates are robustness controls, not held-out accuracy"
        ),
        "source_used_by_matcher": False,
        "pitch_gate": args.pitch_gate,
        "sequence_gate": args.sequence_gate,
        "interpolated_positions": args.interpolated_positions,
        "reference": (
            "Existing count-exact strictly monotonic vocal notehead linkage; pixel ownership is not independently certified"
        ),
        "production_qualified": False,
        "ambiguity_margin_page_pixels": 5,
        "totals": totals,
        "original_cases": cases,
        "wrong_matches": errors,
        "input_sha256": hashes,
        "code_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("lyric_notehead_association.py"),
                Path(__file__).with_name("lyric_staff_pitch_geometry.py"),
                Path(__file__).with_name("lyric_row_routing.py"),
            ]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, report)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
