"""Inventory real count differences and stress repeated-pitch ownership separately."""

import argparse
import json
from collections import Counter
from pathlib import Path

from training.onnx.lyric_notehead_association import (
    associate_x,
    pitch_staff_position,
    reject_pitch_conflicts,
    reject_sequence_ambiguity,
)
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--linkage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[20, 40, 80, 120])
    parser.add_argument("--sequence-gate", action="store_true")
    args = parser.parse_args()
    pages = json.loads(args.linkage.read_text())["pages"]
    hashes = {str(args.linkage): sha(args.linkage)}
    inventory, challenges = [], []
    by_offset = {}
    summary = {"cases": 0, "matched": 0, "correct": 0, "wrong": 0, "wrong_same_pitch": 0}
    for page_index, page in enumerate(pages):
        folder = args.geometry / f"page-{page_index}"
        paths = [
            folder / name
            for name in (
                "staff-transforms.json",
                "note-anchors.json",
                "final-note-anchors.json",
                "decoded-staves.json",
            )
        ]
        for path in paths:
            hashes[str(path)] = sha(path)
        transforms, first, final, voices = [json.loads(path.read_text()) for path in paths]
        final_counts = Counter(row.get("staff_call") for row in final)
        first_counts = Counter(row["staff_call"] for row in first)
        for transform in transforms:
            call = transform["staff_call"]
            rows = [r for r in final if r.get("staff_call") == call]
            heads = transform["detected_noteheads"]
            mapped = [r for r in rows if r["page_coordinate"] is not None]
            result = associate_x(
                [r["page_coordinate"][0] for r in mapped],
                [h[0] * transform["original_to_processed_ratio"][0] for h in heads],
                max_distance=40,
                ambiguity_margin=5,
            )
            inventory.append(
                {
                    "score": page["score"],
                    "staff_call": call,
                    "staff_lines": len(transform["staff_grid"][0]["y"]),
                    "segmented_heads": len(heads),
                    "first_pass_notes": first_counts[call],
                    "final_notes": final_counts[call],
                    "final_count_mismatch": len(heads) != final_counts[call],
                    "mapped_final": len(mapped),
                    "geometry_pairs": len(result["pairs"]),
                    "geometry_unresolved_decoded": len(result["unresolved_decoded"]),
                    "ownership_accuracy": None,
                }
            )
        clefs = [r["rhythm"] for r in voices[0] if r["rhythm"].startswith("clef_")]
        if len(clefs) != 1:
            raise ValueError("Changing vocal clefs unsupported")
        for system in page["systems"]:
            links = system["proposed_links"]
            if not links:
                continue
            transform = next(t for t in transforms if t["staff_call"] == system["staff_call"])
            heads = sorted(transform["detected_notehead_positions"], key=lambda h: h["center"][0])
            if not system["exact_written_pitch_duration_sequence"] or len(heads) != len(links):
                raise ValueError("Reference mapping is not count exact")
            if args.sequence_gate and (
                any(len(point["y"]) != 5 for point in transform["staff_grid"])
                or any(
                    left["center"][0] >= right["center"][0] for left, right in zip(heads, heads[1:])
                )
            ):
                raise ValueError("Sequence gate requires a single staff and strictly ordered heads")
            decoded = [link["prediction"]["page_coordinate"][0] for link in links]
            detected = [h["center"][0] * transform["original_to_processed_ratio"][0] for h in heads]
            positions = [
                pitch_staff_position(link["prediction"]["pitch"], clefs[0]) for link in links
            ]
            detected_positions = [h["staff_position"] for h in heads]
            # Two explicit synthetic faults: remove one repeated pitch and shift
            # the neighboring same-pitch attention x by a bounded page-pixel offset.
            for missing in range(len(links)):
                for shifted in range(len(links)):
                    if shifted == missing or positions[shifted] != positions[missing]:
                        continue
                    if abs(shifted - missing) > 2:
                        continue
                    indices = [i for i in range(len(links)) if i != missing]
                    for offset in sorted(
                        {sign * offset for offset in args.offsets for sign in (-1, 1)}
                    ):
                        xs = [decoded[i] + (offset if i == shifted else 0) for i in indices]
                        result = reject_pitch_conflicts(
                            associate_x(xs, detected, max_distance=40, ambiguity_margin=5),
                            [positions[i] for i in indices],
                            detected_positions,
                        )
                        if args.sequence_gate:
                            result = reject_sequence_ambiguity(
                                result, [positions[i] for i in indices], detected_positions
                            )
                        wrong = [(indices[i], j) for i, j in result["pairs"] if indices[i] != j]
                        summary["cases"] += 1
                        summary["matched"] += len(result["pairs"])
                        summary["correct"] += len(result["pairs"]) - len(wrong)
                        summary["wrong"] += len(wrong)
                        summary["wrong_same_pitch"] += sum(
                            positions[i] == positions[j] for i, j in wrong
                        )
                        subtotal = by_offset.setdefault(
                            str(abs(offset)), {"cases": 0, "matched": 0, "correct": 0, "wrong": 0}
                        )
                        subtotal["cases"] += 1
                        subtotal["matched"] += len(result["pairs"])
                        subtotal["correct"] += len(result["pairs"]) - len(wrong)
                        subtotal["wrong"] += len(wrong)
                        for i, j in wrong:
                            challenges.append(
                                {
                                    "score": page["score"],
                                    "system": system["scan_system"],
                                    "missing_index": missing,
                                    "shifted_index": shifted,
                                    "offset_pixels": offset,
                                    "decoded_index": i,
                                    "assigned_index": j,
                                    "expected_source_note": links[i]["source"],
                                    "assigned_source_note": links[j]["source"],
                                }
                            )
    report = {
        "scope": (
            "Real count differences are unscored inventories; repeated-pitch perturbations are synthetic calibration stress controls"
        ),
        "production_qualified": False,
        "sequence_gate": args.sequence_gate,
        "attention_offset_magnitudes_page_pixels": args.offsets,
        "source_used_by_matcher": False,
        "inventory": inventory,
        "repeated_pitch_controls": summary,
        "by_attention_offset_magnitude": by_offset,
        "wrong_matches": challenges,
        "input_sha256": hashes,
        "code_sha256": {
            str(path): sha(path)
            for path in (Path(__file__), Path(__file__).with_name("lyric_notehead_association.py"))
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, report)
    print(json.dumps(summary, indent=2))
    print("Real final-count mismatches:", sum(r["final_count_mismatch"] for r in inventory))


if __name__ == "__main__":
    main()
