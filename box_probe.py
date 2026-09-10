"""Do OLiMPiC's annotated system boxes bound the whole system, or only the piano?

A Lieder system is voice staff, lyrics, then piano grand staff. If the boxes cover only
the piano, the scanned images contain no lyrics and no vocal line - and recovering them
means extending the boxes upward rather than re-annotating.

The test: compare each box's height against the distance to the next system on the page.
A box that covers the whole system leaves only the inter-system margin; one that covers
half of it leaves a gap as large as itself.
"""
import statistics
from pathlib import Path

import yaml

root = Path("/workspace/b0/olimpic-probe/imslp_systems")
ratios, heights, pitches = [], [], []
for path in sorted(root.glob("*.yaml")):
    data = yaml.safe_load(path.read_text()) or {}
    for page in (data.get("pages") or {}).values():
        systems = page.get("systems") or []
        boxes = [s["boundingBox"] for s in systems if "boundingBox" in s]
        boxes.sort(key=lambda b: b["top"])
        for first, second in zip(boxes, boxes[1:]):
            pitch = second["top"] - first["top"]
            if pitch <= 0:
                continue
            ratios.append(first["height"] / pitch)
            heights.append(first["height"])
            pitches.append(pitch)

if ratios:
    ratios.sort()
    print(f"{len(ratios):,} system pairs across {len(list(root.glob('*.yaml')))} documents")
    print(f"  median box height      {statistics.median(heights):.0f} px")
    print(f"  median system pitch    {statistics.median(pitches):.0f} px")
    print(f"  median coverage        {statistics.median(ratios):.0%} of the pitch")
    print(f"  quartiles              {ratios[len(ratios)//4]:.0%} / {ratios[3*len(ratios)//4]:.0%}")
    print()
    print("  a box covering the whole system would leave only a small margin (~80-90%);")
    print("  one covering only the piano leaves a gap about as large as itself (~50%).")
else:
    print("no boxes found")
