"""Do the staff boxes of a system overlap, so a crop contains a neighbour's music?

Found while chasing a case a corpus agent flagged as "a piano crop with the vocal line
merged into the upper staff". IMSLP154060 system 0 has a vocal box spanning rows 579-849
and a piano box starting at 735 - a 114px overlap, so the piano crop really does contain
the bottom of the vocal staff.

That is a labelling hazard in a direction nothing has looked at: the crop shows music the
label does not mention, which teaches the model to ignore ink it can see. It is measured
from the detection geometry alone - no alignment, no model, no circularity.
"""
import json
from collections import Counter
from pathlib import Path

import yaml

root = Path("/workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes")
overlaps = []
systems = 0
multi = 0
for path in sorted(root.glob("*.yaml")):
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        continue
    for page_no, page in (doc.get("pages") or {}).items():
        for idx, system in enumerate(page.get("systems", []) or []):
            boxes = system.get("staffBoxes", []) or []
            systems += 1
            if len(boxes) < 2:
                continue
            multi += 1
            ordered = sorted(boxes, key=lambda b: b["top"])
            for a, b in zip(ordered, ordered[1:]):
                gap = b["top"] - (a["top"] + a["height"])
                if gap < 0:
                    overlaps.append({
                        "score": path.stem, "page": page_no, "system": idx,
                        "overlap_px": -gap,
                        "upper_h": a["height"], "lower_h": b["height"],
                        "frac_of_upper": round(-gap / max(a["height"], 1), 3),
                    })

print(f"{systems:,} detected systems, {multi:,} with two or more staff boxes")
print(f"overlapping box pairs: {len(overlaps):,}  "
      f"({100*len(overlaps)/max(multi,1):.1f}% of multi-box systems)")
if overlaps:
    fr = sorted(o["frac_of_upper"] for o in overlaps)
    px = sorted(o["overlap_px"] for o in overlaps)
    print(f"  overlap in px    : median {px[len(px)//2]}, p90 {px[int(0.9*len(px))]}, max {px[-1]}")
    print(f"  as a fraction of the upper box height: median {fr[len(fr)//2]:.2f}, "
          f"p90 {fr[int(0.9*len(fr))]:.2f}, max {fr[-1]:.2f}")
    scores = Counter(o["score"] for o in overlaps)
    print(f"  scores affected  : {len(scores)}")
    print("  worst:")
    for o in sorted(overlaps, key=lambda o: -o["frac_of_upper"])[:6]:
        print(f"    {o['score']}-sys{o['system']}: {o['overlap_px']}px "
              f"= {100*o['frac_of_upper']:.0f}% of the upper staff box")
Path("/workspace/b0/lieder-rebuild/box_overlaps.json").write_text(json.dumps(overlaps, indent=2))
print(f"\nwrote box_overlaps.json")
