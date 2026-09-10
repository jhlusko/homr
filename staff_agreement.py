"""How often will the crop guard fire?

convert_ossq converts a system only when its staff crops number exactly as many as the
MusicXML parts, because the crop-to-part pairing is positional. This reads the staff
detections already on disk and compares them against the part counts, so the drop rate is
known before a training run is spent on what survives.

Counting detections above the confidence threshold without merging overlapping boxes, so
this is an upper bound on how many crops each system will produce - and therefore a lower
bound on how often over-detection causes a mismatch.
"""
import collections, sys, xml.etree.ElementTree as ET
from pathlib import Path

root = Path(sys.argv[1])
threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7

matched = mismatch = 0
detected_by_parts = collections.Counter()
delta = collections.Counter()
for bbox in root.glob("scores/*/*/images/synthetic/systemwise/*_yolo_bboxs.txt"):
    stem = bbox.name.removesuffix("_yolo_bboxs.txt")
    segment = bbox.parents[3] / "musicxml" / "unaligned" / f"{stem}.musicxml"
    if not segment.is_file():
        continue
    try:
        parts = len(ET.parse(segment).getroot().findall("part"))
    except ET.ParseError:
        continue
    staves = sum(
        1 for line in bbox.read_text().splitlines()
        if line.strip() and float(line.split()[-1]) >= threshold
    )
    detected_by_parts[(parts, staves)] += 1
    delta[staves - parts] += 1
    if staves == parts:
        matched += 1
    else:
        mismatch += 1

total = matched + mismatch
print(f"systems compared: {total:,}")
print(f"  detections match the part count: {matched:,} ({matched / max(total,1):.1%})")
print(f"  mismatch (system would be skipped): {mismatch:,} ({mismatch / max(total,1):.1%})")
print("\ndetected minus parts:")
for d, count in sorted(delta.items()):
    print(f"  {d:+d}: {count:,} ({count / max(total,1):.1%})")
print("\nmost common (parts, detected) pairs:")
for (parts, staves), count in detected_by_parts.most_common(8):
    print(f"  parts={parts} detected={staves}: {count:,}")
