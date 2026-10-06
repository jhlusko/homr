"""Retain detected but undecoded physical staffs as routing boundaries only."""

from training.onnx.lyric_row_routing import lines_at, physical_staffs


def complete_routing_transforms(runtime, raw):
    present = physical_staffs(runtime)
    detected = physical_staffs(raw)
    matched, additions = set(), []
    for index, staff in enumerate(detected):
        candidates = []
        for j, existing in enumerate(present):
            left = max(staff["grid"][0][0], existing["grid"][0][0])
            right = min(staff["grid"][-1][0], existing["grid"][-1][0])
            if right <= left:
                continue
            deltas = []
            spacings = []
            for x in [left, (left + right) / 2, right]:
                a, b = lines_at(staff, x), lines_at(existing, x)
                deltas.extend(abs(y - z) for y, z in zip(a, b))
                spacings.append((a[-1] - a[0]) / 4)
            if min(spacings) > 0 and max(deltas) <= min(spacings) / 4:
                candidates.append(j)
        if len(candidates) > 1 or (candidates and candidates[0] in matched):
            raise ValueError("Physical staff correspondence is ambiguous")
        if candidates:
            matched.add(candidates[0])
        else:
            additions.append(
                {
                    "staff_call": -1000 - index,
                    "staff_grid": [{"x": x, "y": ys} for x, ys in staff["grid"]],
                    "original_to_processed_ratio": [1, 1],
                    "routing_boundary_only": True,
                    "decoded_note_ownership": False,
                }
            )
    if len(matched) != len(present):
        raise ValueError("Complete geometry does not reproduce every runtime staff")
    return [*runtime, *additions], {
        "added_boundaries": len(additions),
        "runtime_physical_staffs": len(present),
        "detected_physical_staffs": len(detected),
        "match_tolerance_staff_spacing_fraction": 0.25,
        "native_input_used": False,
    }
