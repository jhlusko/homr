"""Diagnostic recovery of rejected OCR boxes in existing selected physical rows."""

import argparse
import json
import math
from pathlib import Path
from statistics import median

from training.onnx.lyric_complete_staff_context import complete_routing_transforms
from training.onnx.lyric_row_routing import lines_at, physical_staffs, route_rows
from training.onnx.lyric_vocal_context import select_vocal_voice
from training.onnx.qualify_joint_export import sha, write


def box_key(row):
    return tuple(tuple(point) for point in row["box"])


def vocal_calls_from_final(final, voice=0):
    """Untraced final notes cannot establish staff ownership."""
    return {
        r["staff_call"]
        for r in final
        if r["voice_index"] == voice and r.get("staff_call") is not None
    }


def recover(selected, all_rows, transforms, vocal_calls):
    """Use frozen selected rows as seeds, never native labels or recovered seeds.

    Candidates must fit an unpadded physical gap and overlap every member of
    exactly one existing vocal row by half the larger box height. Alphabetic
    content is required, but this is not a semantic lyric-role classifier.
    """
    baseline = route_rows(selected, transforms, recover_row_neighbors=True)
    staffs = physical_staffs(transforms)
    groups = {}
    for route in baseline["routed"]:
        if route["staff_call"] in vocal_calls and "physical_row_order" in route:
            key = (route["physical_staff_index"], route["physical_row_order"])
            groups.setdefault(key, []).append(route)
    selected_keys = {box_key(row) for row in selected}
    all_by_box = {box_key(row): row for row in all_rows}
    if len(all_by_box) != len(all_rows):
        raise ValueError("Duplicate full-OCR boxes")
    if any(all_by_box.get(box_key(row), {}).get("text") != row["text"] for row in selected):
        raise ValueError("Selected OCR rows differ from full cached rows")
    accepted, rejected = [], []
    for index, row in enumerate(all_rows):
        if box_key(row) in selected_keys:
            continue
        record = {"full_ocr_index": index, "text": row["text"], "box": row["box"]}
        xs, ys = zip(*row["box"])
        if not any(c.isalpha() for c in row["text"]):
            rejected.append({**record, "reason": "no_alphabetic_content"})
            continue
        if not all(math.isfinite(v) for v in [*xs, *ys]) or max(ys) <= min(ys):
            rejected.append({**record, "reason": "invalid_box"})
            continue
        x, low, high = (min(xs) + max(xs)) / 2, min(ys), max(ys)
        eligible = []
        for (staff_index, order), seeds in groups.items():
            if staff_index + 1 >= len(staffs):
                continue
            lines = lines_at(staffs[staff_index], x)
            following = lines_at(staffs[staff_index + 1], x)
            if lines is None or following is None or low <= lines[-1] or high >= following[0]:
                continue
            # Bounded extrapolation: distant directions must not inherit a row.
            # Four median seed heights is an explicit diagnostic gate, uncalibrated.
            seed_boxes = [selected[seed["index"]]["box"] for seed in seeds]
            seed_xs = [point[0] for box in seed_boxes for point in box]
            extension = 4 * median(seed["bounds_y"][1] - seed["bounds_y"][0] for seed in seeds)
            if min(xs) < min(seed_xs) - extension or max(xs) > max(seed_xs) + extension:
                continue
            if all(
                min(high, seed["bounds_y"][1]) - max(low, seed["bounds_y"][0])
                >= max(high - low, seed["bounds_y"][1] - seed["bounds_y"][0]) / 2
                for seed in seeds
            ):
                eligible.append((staff_index, order))
        if len(eligible) != 1:
            rejected.append(
                {**record, "reason": "no_unique_established_row", "candidate_rows": len(eligible)}
            )
            continue
        staff_index, order = eligible[0]
        accepted.append(
            {
                **record,
                "selected_index": len(selected) + len(accepted),
                "physical_staff_index": staff_index,
                "physical_row_order": order,
                "staff_call": staffs[staff_index]["staff_call"],
            }
        )
    rows = [*selected, *(all_rows[r["full_ocr_index"]] for r in accepted)]
    combined = {
        r["index"]: r for r in route_rows(rows, transforms, recover_row_neighbors=True)["routed"]
    }
    for old in baseline["routed"]:
        new = combined.get(old["index"], {})
        if any(
            new.get(k) != old.get(k) for k in ["staff_call", "staff_group", "physical_row_order"]
        ):
            raise ValueError("Recovery changed an existing physical row")
    for record in accepted:
        route = combined.get(record["selected_index"], {})
        if any(
            route.get(k) != record[k]
            for k in ["staff_call", "physical_row_order", "physical_staff_index"]
        ):
            raise ValueError("Recovered box changed its established row")
    return rows, {"accepted": accepted, "rejected": rejected, "native_input_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pages", "selected", "all-ocr", "decode", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--vocal-context", action="store_true")
    parser.add_argument("--physical-context", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pages = json.loads(args.pages.read_text())
    hashes, results = {str(args.pages): sha(args.pages)}, []
    for index, page in enumerate(pages):
        paths = [
            args.selected / f"{page['score']}.json",
            args.all_ocr / f"{page['score']}.json",
            args.decode / f"page-{index}" / "staff-transforms.json",
            args.decode / f"page-{index}" / "final-note-anchors.json",
            args.decode / f"page-{index}" / "identity.json",
        ]
        selected, full, transforms, final, identity = [json.loads(p.read_text()) for p in paths]
        if any(
            r["image_sha256"] != page["image_sha256"] for r in [selected, full, identity["page"]]
        ):
            raise ValueError("Page image hashes differ")
        if args.physical_context:
            raw_folder = args.physical_context / f"page-{index}"
            route_path = raw_folder / "routing.json"
            if json.loads(route_path.read_text())["page"]["image_sha256"] != page["image_sha256"]:
                raise ValueError("Physical boundary context image differs")
            raw_path = raw_folder / "staff-transforms.json"
            transforms, _ = complete_routing_transforms(
                transforms, json.loads(raw_path.read_text())
            )
            paths.extend([route_path, raw_path])
        voice = (
            select_vocal_voice(
                final,
                transforms,
                route_rows(selected["all_page_text"], transforms, recover_row_neighbors=True),
            )["voice_index"]
            if args.vocal_context
            else 0
        )
        calls = vocal_calls_from_final(final, voice)
        rows, evidence = recover(
            selected["all_page_text"], full["all_page_text"], transforms, calls
        )
        write(
            args.out / f"{page['score']}.json",
            {
                **selected,
                "all_page_text": rows,
                "selection": "baseline_plus_existing_physical_row_recovery_control",
                "row_recovery": evidence,
            },
        )
        results.append(
            {
                "score": page["score"],
                "accepted": len(evidence["accepted"]),
                "rejected": len(evidence["rejected"]),
            }
        )
        hashes.update({str(p): sha(p) for p in paths})
    write(
        args.out / "receipt.json",
        {
            "pages": results,
            "input_sha256": hashes,
            "native_input_used": False,
            "production_qualified": False,
            "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
