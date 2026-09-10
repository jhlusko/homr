"""How often does a slur cross a system break in the real segments?

If it is common, dropping unmatched stops mislabels every one of those notes as carrying
no slur, on crops that plainly show a slur ending.
"""
import sys, xml.etree.ElementTree as ET
from pathlib import Path
from training.omr_datasets.structured_notation_parser import parse_part, NotationExtractor, Findings

root = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 400

totals = Findings()
segments = 0
segments_with_orphans = 0
for path in sorted(root.glob("scores/*/*/musicxml/unaligned/*.musicxml"))[:limit]:
    try:
        tree = ET.parse(path).getroot()
    except ET.ParseError:
        continue
    segments += 1
    before = totals.unmatched_stops
    for part in tree.findall("part"):
        _, findings = parse_part(part)
        for field in vars(findings):
            setattr(totals, field, getattr(totals, field) + getattr(findings, field))
    if totals.unmatched_stops > before:
        segments_with_orphans += 1

print(f"segments examined: {segments}")
for field, value in vars(totals).items():
    print(f"  {field}: {value:,}")
print(f"segments containing at least one unmatched stop: {segments_with_orphans}"
      f" ({segments_with_orphans / max(segments,1):.1%})")
