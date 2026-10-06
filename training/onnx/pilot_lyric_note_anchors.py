"""Capture exact staff transforms and attention coordinates on calibration scans.

Diagnostic instrumentation only. No reference text/note locations enter inference.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.onnx.lyric_anchor_geometry import canvas_to_page
from training.onnx.qualify_joint_export import setup, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--page-index", type=int, required=True)
    parser.add_argument("--arm", default="candidate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Pilot output must be new")
    page = json.loads(args.pages.read_text())[args.page_index]
    if page["partition"] != "calibration":
        raise ValueError("Only frozen calibration pages are permitted")
    setup()
    from homr import staff_parsing as parsing
    from training.onnx import qualify_joint_pages as runner

    runner.setup = lambda: None
    original_size = np.array(cv2.imread(page["image"]).shape[:2][::-1])
    transforms, anchors, final_anchors = [], [], []
    active = {}
    original_prepare = parsing.prepare_staff_image
    original_crop = parsing.crop_image_and_return_new_top
    original_dewarp = parsing.dewarp_staff_image
    original_center = parsing.center_image_on_canvas

    def capture_crop(image, *bounds):
        result = original_crop(image, *bounds)
        if active:
            active["crop_origins"].append(result[1].tolist())
        return result

    def capture_dewarp(image, staff, index, debug):
        result = original_dewarp(image, staff, index, debug)
        tform = result.tform
        active["dewarp"] = dict(
            source=None if tform is None else tform.src_points.tolist(),
            destination=None if tform is None else tform.dst_points.tolist(),
            simplices=[] if tform is None else tform.triangulation.simplices.tolist(),
        )
        return result

    def capture_center(image, size, margin_top=0, margin_bottom=0):
        result = original_center(image, size, margin_top, margin_bottom)
        active["canvas"] = dict(
            input_size=list(image.shape[:2][::-1]),
            resized_size=size.tolist(),
            offset=[
                0,
                (result.shape[0] - margin_top - margin_bottom - int(size[1])) // 2 + margin_top,
            ],
        )
        return result

    def capture_prepare(debug, index, staff, image, regions):
        active.clear()
        region = parsing._calculate_region(staff, regions)
        size = parsing.get_tr_omr_canvas_size(
            (int(region[3] - region[1]), int(region[2] - region[0]))
        )
        active.update(
            staff_call=index,
            crop_origins=[],
            staff_scale=float(size[1] / (region[3] - region[1])),
            original_to_processed_ratio=(original_size / np.array(image.shape[:2][::-1])).tolist(),
            detected_noteheads=[
                list(symbol.center) for symbol in staff.symbols if type(symbol).__name__ == "Note"
            ],
        )
        result = original_prepare(debug, index, staff, image, regions)
        transforms.append(json.loads(json.dumps(active)))
        return result

    def save_symbols(index, symbols):
        transform = next(t for t in reversed(transforms) if t["staff_call"] == index)
        for token_index, symbol in enumerate(symbols):
            if not symbol.rhythm.startswith("note_"):
                continue
            row = dict(
                staff_call=index,
                token_index=token_index,
                rhythm=symbol.rhythm,
                pitch=symbol.pitch,
                position=symbol.position,
                page_coordinate=None,
            )
            try:
                coordinate = np.asarray(symbol.coordinates, dtype=float).reshape(-1)
                row["canvas_coordinate"] = coordinate.tolist()
                row["page_coordinate"] = canvas_to_page(coordinate, transform)
            except (ValueError, TypeError) as error:
                row["geometry_skip"] = str(error)
            anchors.append(row)
            # Diagnostic-only metadata follows copy.copy/deepcopy repairs without
            # participating in token identity or influencing inference.
            symbol._lyric_anchor_provenance = dict(row)

    original_plain = parsing.parse_staff_image
    original_greedy = parsing.parse_staff_image_greedy_with_margins
    original_final = parsing.parse_staffs

    def capture_plain(*pos, **kw):
        result = original_plain(*pos, **kw)
        save_symbols(pos[1], result)
        return result

    def capture_greedy(*pos, **kw):
        result = original_greedy(*pos, **kw)
        save_symbols(pos[1], result[0])
        return result

    def capture_final(*pos, **kw):
        result = original_final(*pos, **kw)
        for voice_index, symbols in enumerate(result):
            system_index = 0
            for token_index, symbol in enumerate(symbols):
                if symbol.rhythm == "newline":
                    system_index += 1
                if not symbol.rhythm.startswith("note_"):
                    continue
                origin = getattr(symbol, "_lyric_anchor_provenance", None)
                row = dict(
                    voice_index=voice_index,
                    system_index=system_index,
                    token_index=token_index,
                    rhythm=symbol.rhythm,
                    pitch=symbol.pitch,
                    position=symbol.position,
                    page_coordinate=None,
                    first_pass_origin=origin,
                )
                if origin is None:
                    row["geometry_skip"] = "No retained first-pass provenance"
                else:
                    transform = next(
                        t for t in transforms if t["staff_call"] == origin["staff_call"]
                    )
                    row["staff_call"] = origin["staff_call"]
                    row["rhythm_changed"] = symbol.rhythm != origin["rhythm"]
                    row["pitch_changed"] = symbol.pitch != origin["pitch"]
                    try:
                        coordinate = np.asarray(symbol.coordinates, dtype=float).reshape(-1)
                        row["canvas_coordinate"] = coordinate.tolist()
                        row["page_coordinate"] = canvas_to_page(coordinate, transform)
                    except (ValueError, TypeError) as error:
                        row["geometry_skip"] = str(error)
                final_anchors.append(row)
        return result

    parsing.prepare_staff_image = capture_prepare
    parsing.crop_image_and_return_new_top = capture_crop
    parsing.dewarp_staff_image = capture_dewarp
    parsing.center_image_on_canvas = capture_center
    parsing.parse_staff_image = capture_plain
    parsing.parse_staff_image_greedy_with_margins = capture_greedy
    parsing.parse_staffs = capture_final
    try:
        runner.process(page, json.loads(args.models.read_text()), args.arm, args.out)
    finally:
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out / "staff-transforms.json", transforms)
        write(args.out / "note-anchors.json", anchors)
        write(args.out / "final-note-anchors.json", final_anchors)
        write(
            args.out / "anchor-receipt.json",
            dict(
                pages_sha256=sha(args.pages),
                models_sha256=sha(args.models),
                instrumentation_sha256=sha(Path(__file__)),
                geometry_sha256=sha(Path(__file__).with_name("lyric_anchor_geometry.py")),
                anchors=len(anchors),
                mapped=sum(row["page_coordinate"] is not None for row in anchors),
                final_notes=len(final_anchors),
                final_mapped=sum(row["page_coordinate"] is not None for row in final_anchors),
                source_geometry_injected=False,
                source_ownership_certified=False,
                scope="Diagnostic first-pass and final-decoded anchor provenance; native ownership and final XML linkage require independent checks",
            ),
        )


if __name__ == "__main__":
    main()
