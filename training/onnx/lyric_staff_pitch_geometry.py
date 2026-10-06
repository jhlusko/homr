"""Diagnostic whole-staff pitch geometry; stored detector positions remain unchanged."""

import math

from training.onnx.lyric_row_routing import lines_at, physical_staffs


def interpolate_notehead_positions(transform):
    staffs = physical_staffs([transform])
    if len(staffs) != 1:
        raise ValueError("Pitch diagnostic requires one five-line staff")
    ratio_x, ratio_y = transform["original_to_processed_ratio"]
    records = []
    for head in sorted(transform["detected_notehead_positions"], key=lambda h: h["center"][0]):
        x, y = head["center"][0] * ratio_x, head["center"][1] * ratio_y
        lines = lines_at(staffs[0], x)
        if (
            lines is None
            or not all(math.isfinite(value) for value in [x, y, *lines])
            or lines[-1] <= lines[0]
        ):
            raise ValueError("Unusable local staff geometry")
        continuous = 1 + 8 * (lines[-1] - y) / (lines[-1] - lines[0])
        records.append(
            {
                "center": head["center"],
                "stored_staff_position": head["staff_position"],
                "continuous_position": continuous,
                "staff_position": round(continuous),
                "rounding_residual": abs(continuous - round(continuous)),
            }
        )
    return records
