"""What does the empty-final-measure filter cost, and what does it catch?

Run on the files that already pass PDMX's own pre-filters, so the number is the marginal
cost of this check rather than of the whole selection.
"""
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter

from training.omr_datasets.convert_pdmx import _load_filtered_paths, _read_mxl

sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
paths = _load_filtered_paths()
print(f"{len(paths):,} files pass PDMX's existing pre-filters")

random.Random(0).shuffle(paths)
counts = Counter()
for path in paths[:sample_size]:
    try:
        parts = ET.fromstring(_read_mxl(path)).findall("part")
    except Exception:
        counts["unreadable"] += 1
        continue
    if not parts:
        counts["no parts"] += 1
        continue
    verdict = "kept"
    for part in parts:
        measures = part.findall("measure")
        if not measures:
            continue
        notes = measures[-1].findall("note")
        if not notes:
            verdict = "final bar has no notes"
            break
        if all(n.find("rest") is not None for n in notes):
            verdict = "final bar is all rests"
            break
    counts[verdict] += 1

total = sum(counts.values())
for name, count in counts.most_common():
    print(f"  {name}: {count:,} ({count / total:.1%})")
