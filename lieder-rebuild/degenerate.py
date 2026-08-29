"""Detected 'systems' that cannot be music, corpus-wide.

Found while testing whether displacement onsets coincide with phantom systems - they do
not, but IMSLP280033 carries a detected system 0.022 of the page wide with zero
barlines and one staff box. That is not a system; it is a stray mark the detector
promoted. Every such entry shifts the index of every system after it on its page, and
both alignment methods consume the same detected list, so consensus cannot see it.
"""
import json
from collections import defaultdict

rows = json.load(open("/workspace/b0/lieder-rebuild/bar_count_rows_v2.json"))
by = defaultdict(list)
for r in rows:
    by[r["score_id"]].append(r)

degenerate = []
for sid, rs in by.items():
    widths = sorted(r["system_width_fraction"] for r in rs)
    med = widths[len(widths) // 2] if widths else 0
    for r in rs:
        reasons = []
        if med and r["system_width_fraction"] < 0.35 * med:
            reasons.append(f"width {r['system_width_fraction']:.3f} vs median {med:.3f}")
        if r["detected"] == 0:
            reasons.append("no barlines detected")
        if r["staff_box_count"] == 0:
            reasons.append("no staff boxes")
        if len(reasons) >= 2:
            degenerate.append({"score_id": sid, "page_index": r["page_index"],
                               "system_index": r["system_index"],
                               "width": round(r["system_width_fraction"], 4),
                               "barlines": r["detected"],
                               "staves": r["staff_box_count"],
                               "why": "; ".join(reasons)})

print(f"{len(rows):,} detected systems across {len(by)} scores")
print(f"{len(degenerate)} are degenerate (two or more independent signs of not being a system)")
print(f"affecting {len({d['score_id'] for d in degenerate})} scores\n")
for d in sorted(degenerate, key=lambda d: (d["score_id"], d["page_index"], d["system_index"]))[:25]:
    print(f"  {d['score_id']:>14} p{d['page_index']:<3} sys{d['system_index']:<3} "
          f"width {d['width']:<7} barlines {d['barlines']:<3} staves {d['staves']}   {d['why']}")
json.dump(degenerate, open("/workspace/b0/lieder-rebuild/degenerate_systems.json", "w"), indent=2)
print(f"\nwrote degenerate_systems.json")
