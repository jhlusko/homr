"""The same distribution, counting sounding notes only - the definition actually used."""
import random
import statistics
import xml.etree.ElementTree as ET

from training.omr_datasets.convert_pdmx import _load_filtered_paths, _read_mxl, sounding_notes

paths = _load_filtered_paths()
random.Random(0).shuffle(paths)
counts = []
for path in paths[:1200]:
    try:
        root = ET.fromstring(_read_mxl(path))
    except Exception:
        continue
    counts.append(sounding_notes(root.findall("part")))

counts.sort()
print(f"{len(counts):,} scores, sounding notes only")
print(f"  min {counts[0]}  median {statistics.median(counts):.0f}  mean {statistics.mean(counts):.0f}")
total = sum(counts)
for threshold in (32, 48, 64, 96):
    lost = sum(1 for n in counts if n < threshold)
    kept = sum(n for n in counts if n >= threshold)
    print(f"  threshold {threshold:>3}: drops {lost:>4} scores ({lost/len(counts):5.1%}), "
          f"keeps {kept/total:6.2%} of sounding notes")
