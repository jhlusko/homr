"""Audit geometry-only matching using count-exact calibration and simulated omissions."""

import argparse
import json
from pathlib import Path

from training.onnx.lyric_notehead_association import associate_x
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    linkage_path = args.anchors / "source-linkage.json"
    pages = json.loads(linkage_path.read_text())["pages"]
    totals = {}
    cases = []
    errors = []
    hashes = {str(linkage_path): sha(linkage_path)}
    for page_index, page in enumerate(pages):
        transform_path = args.anchors / f"page-{page_index}/staff-transforms.json"
        hashes[str(transform_path)] = sha(transform_path)
        transforms = json.loads(transform_path.read_text())
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
            for p in [Path(__file__), Path(__file__).with_name("lyric_notehead_association.py")]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, report)
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
