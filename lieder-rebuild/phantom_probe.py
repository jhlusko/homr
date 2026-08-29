"""Do the displacement onsets coincide with an anomalous detected system?

The rebuilt labels are displaced from a known system onward in 10 scores, found by
comparing against the pre-rebuild corpus. If a phantom system is the cause, the system
at or just before the onset should look unlike its neighbours - narrower, fewer
barlines, or a different staff count.

If that holds, it is a detector: phantoms can be found from geometry alone, with no
model and no second corpus to diff against.
"""
import json
from collections import defaultdict

rows = json.load(open("/workspace/b0/lieder-rebuild/bar_count_rows_v2.json"))
by = defaultdict(list)
for r in rows:
    by[r["score_id"]].append(r)

onsets = {"IMSLP122258": 7, "IMSLP280033": 4, "IMSLP558713": 6,
          "IMSLP624193": 6, "IMSLP634834": 13, "IMSLP83314": 10}

for sid, onset in sorted(onsets.items()):
    rs = sorted(by.get(sid, []), key=lambda r: (r["page_index"], r["system_index"]))
    if not rs:
        print(f"{sid}: no rows")
        continue
    widths = sorted(r["system_width_fraction"] for r in rs)
    med = widths[len(widths) // 2]
    bars = sorted(r["detected"] for r in rs)
    med_bars = bars[len(bars) // 2]
    print(f"{sid}: {len(rs)} systems, median width {med:.3f}, median barlines {med_bars}, onset ~{onset}")
    for i in range(max(0, onset - 3), min(len(rs), onset + 2)):
        r = rs[i]
        flags = []
        if r["system_width_fraction"] < 0.6 * med:
            flags.append("NARROW")
        if r["detected"] <= max(1, med_bars // 3):
            flags.append("FEW-BARLINES")
        marker = " <<<" if i == onset else ""
        print("    idx {:3d}  width {:.3f}  barlines {:2d}  staves {}  {}{}".format(
            i, r["system_width_fraction"], r["detected"], r["staff_box_count"],
            " ".join(flags), marker))
