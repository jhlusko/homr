"""Did the existing narrow-detection filter actually catch the degenerate systems?

align_lieder_systems zeroes a detection narrower than min_width_fraction so it cannot
consume score music. That threshold is an ABSOLUTE fraction of page width, while
"degenerate" is only meaningful relative to the score's own systems - IMSLP454110's
median system is 0.484 of the page wide, so an absolute threshold tuned for a 0.9
median either misses its stray marks or eats its real systems.
"""
import json
from collections import Counter

degen = json.load(open("/workspace/b0/lieder-rebuild/degenerate_systems.json"))
align = json.load(open("/workspace/b0/lieder-rebuild/system_alignment_v2.json"))["scores"]

key = {(d["score_id"], d["page_index"], d["system_index"]) for d in degen}
status = Counter()
consuming = []
for sid, rec in align.items():
    for item in rec.get("systems", []):
        k = (sid, item.get("page_index"), item.get("system_index"))
        if k not in key:
            continue
        st = item.get("status", "?")
        status[st] += 1
        span = (item.get("end_measure") or 0) - (item.get("start_measure") or 0)
        if st == "aligned" and span > 0:
            consuming.append((sid, item.get("page_index"), item.get("system_index"), span))

print(f"{len(degen)} degenerate systems; status in the alignment:")
for st, n in status.most_common():
    print(f"  {st:>12}: {n}")
print(f"\ndegenerate systems that were ALIGNED and consumed real measures: {len(consuming)}")
for c in consuming[:20]:
    print(f"  {c[0]} p{c[1]} sys{c[2]} -> {c[3]} measures")
