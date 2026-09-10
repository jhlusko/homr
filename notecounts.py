"""How many notes does a PDMX score actually have?

A minimum-notes threshold should come from the distribution, not from taste. Measured over
files that already pass PDMX's own filters, so it is the marginal effect of this check.
"""
import random
import statistics
import xml.etree.ElementTree as ET

from training.omr_datasets.convert_pdmx import _load_filtered_paths, _read_mxl

paths = _load_filtered_paths()
random.Random(0).shuffle(paths)
counts = []
for path in paths[:1200]:
    try:
        root = ET.fromstring(_read_mxl(path))
    except Exception:
        continue
    notes = sum(1 for _ in root.iter("note"))
    measures = max((len(p.findall("measure")) for p in root.findall("part")), default=0)
    counts.append((notes, measures))

counts.sort()
notes = [n for n, _ in counts]
print(f"{len(counts):,} scores sampled")
print(f"  min {notes[0]}  median {statistics.median(notes):.0f}  mean {statistics.mean(notes):.0f}  max {notes[-1]:,}")
print()
for pct in (1, 5, 10, 25, 50):
    idx = max(0, len(notes) * pct // 100 - 1)
    print(f"  {pct:>2}th percentile: {notes[idx]:,} notes")
print()
for threshold in (16, 32, 48, 64, 96, 128, 192, 256):
    lost = sum(1 for n in notes if n < threshold)
    kept_notes = sum(n for n in notes if n >= threshold)
    total_notes = sum(notes)
    print(f"  threshold {threshold:>3}: drops {lost:>4} scores ({lost/len(notes):5.1%}), "
          f"keeps {kept_notes/total_notes:6.2%} of all notes")
