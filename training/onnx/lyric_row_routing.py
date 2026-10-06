"""Diagnostic routing to physical staffs/rows; never invent native verse identity."""

import math


def physical_staffs(transforms):
    staffs = []
    for transform in transforms:
        ratio_x, ratio_y = transform["original_to_processed_ratio"]
        grid = transform["staff_grid"]
        line_counts = {len(point["y"]) for point in grid}
        if not grid or len(line_counts) != 1 or next(iter(line_counts)) % 5:
            raise ValueError("Invalid staff grid")
        for offset in range(0, next(iter(line_counts)), 5):
            staffs.append(
                {
                    "staff_call": transform["staff_call"],
                    "staff_group": offset // 5,
                    "grid": [
                        (p["x"] * ratio_x, [y * ratio_y for y in p["y"][offset : offset + 5]])
                        for p in grid
                    ],
                }
            )
    staffs.sort(key=lambda staff: sum(staff["grid"][0][1]) / 5)
    return staffs


def lines_at(staff, x):
    grid = staff["grid"]
    if x < grid[0][0] or x > grid[-1][0]:
        return None
    for left, right in zip(grid, grid[1:]):
        if left[0] <= x <= right[0]:
            weight = (x - left[0]) / (right[0] - left[0])
            return [a + weight * (b - a) for a, b in zip(left[1], right[1])]
    return grid[0][1] if x == grid[0][0] else None


def route_rows(rows, transforms, *, recover_row_neighbors=False):
    """Assign intact preselected OCR rows to gaps, then physical row order.

    Input selection remains an independent lyric-role diagnostic. This cannot
    distinguish lyrics from dynamics and cannot map physical rows to native verses.
    Entire vertical OCR bounds must fit between staffs; margins use half a line
    spacing. Below the last staff is intentionally unresolved (no lower bound).
    """
    staffs = physical_staffs(transforms)
    routed, unresolved = [], []
    for index, row in enumerate(rows):
        xs, ys = zip(*row["box"])
        if not all(math.isfinite(v) for v in [*xs, *ys]):
            unresolved.append({"index": index, "reason": "nonfinite_box"})
            continue
        x = (min(xs) + max(xs)) / 2
        candidates = []
        for staff_index, (staff, following) in enumerate(zip(staffs, staffs[1:])):
            lines, next_lines = lines_at(staff, x), lines_at(following, x)
            if lines is None or next_lines is None:
                continue
            spacing = (lines[-1] - lines[0]) / 4
            if min(ys) > lines[-1] + spacing / 2 and max(ys) < next_lines[0] - spacing / 2:
                candidates.append((staff_index, staff))
        if len(candidates) != 1:
            unresolved.append(
                {"index": index, "reason": "no_unique_staff_gap", "candidates": len(candidates)}
            )
            continue
        staff_index, staff = candidates[0]
        routed.append(
            {
                "index": index,
                "staff_call": staff["staff_call"],
                "staff_group": staff["staff_group"],
                "physical_staff_index": staff_index,
                "bounds_y": [min(ys), max(ys)],
                "native_verse": None,
            }
        )
    if recover_row_neighbors:
        # Only strict-pass rows may seed recovery, never another recovered row.
        seeds = list(routed)
        remaining = []
        for item in unresolved:
            row = rows[item["index"]]
            xs, ys = zip(*row["box"])
            if not all(math.isfinite(v) for v in [*xs, *ys]):
                remaining.append(item)
                continue
            x, low, high = (min(xs) + max(xs)) / 2, min(ys), max(ys)
            eligible = []
            for staff_index, (staff, following) in enumerate(zip(staffs, staffs[1:])):
                lines, next_lines = lines_at(staff, x), lines_at(following, x)
                if lines is None or next_lines is None or low <= lines[-1] or high >= next_lines[0]:
                    continue
                neighbors = [
                    seed
                    for seed in seeds
                    if seed["physical_staff_index"] == staff_index
                    and min(high, seed["bounds_y"][1]) - max(low, seed["bounds_y"][0])
                    >= min(high - low, seed["bounds_y"][1] - seed["bounds_y"][0]) / 2
                ]
                if neighbors:
                    eligible.append((staff_index, staff, neighbors))
            if len(eligible) != 1:
                remaining.append(item)
                continue
            staff_index, staff, neighbors = eligible[0]
            routed.append(
                {
                    "index": item["index"],
                    "staff_call": staff["staff_call"],
                    "staff_group": staff["staff_group"],
                    "physical_staff_index": staff_index,
                    "bounds_y": [low, high],
                    "native_verse": None,
                    "recovery": "strict_same_row_neighbor_inside_unpadded_staff_gap",
                    "seed_indices": [seed["index"] for seed in neighbors],
                }
            )
        unresolved = remaining
    for staff_index in {row["physical_staff_index"] for row in routed}:
        groups = []
        for row in sorted(
            (r for r in routed if r["physical_staff_index"] == staff_index),
            key=lambda r: sum(r["bounds_y"]),
        ):
            low, high = row["bounds_y"]
            compatible = [
                group
                for group in groups
                if all(
                    min(high, other["bounds_y"][1]) - max(low, other["bounds_y"][0])
                    >= min(high - low, other["bounds_y"][1] - other["bounds_y"][0]) / 2
                    for other in group
                )
            ]
            if len(compatible) == 1:
                compatible[0].append(row)
            elif not compatible:
                groups.append([row])
            else:
                row["row_skip"] = "ambiguous_row_overlap"
        for order, group in enumerate(groups, 1):
            for row in group:
                row["physical_row_order"] = order
    return {"routed": routed, "unresolved": unresolved, "native_verse_identity_inferred": False}
