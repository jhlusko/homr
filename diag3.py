"""Why did slur placement produce nothing?

Walk the exact path convert_ossq takes: locate the whole score, build the index, pull a
slice, and apply it to the extracted part.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.convert_ossq import extract_part
from training.omr_datasets.slur_placement import PlacementIndex, apply_placements

root = Path("/workspace/b0/ossq-omr")
work = next(p for p in sorted((root / "scores").glob("*/*")) if p.is_dir()
            and list(p.glob("*.musicxml")))
segments = sorted((work / "musicxml" / "unaligned").glob("*.musicxml"))
score_id = segments[0].stem.split(":")[0]
whole = work / f"{score_id}.musicxml"
print("work:", work.name)
print("score_id:", score_id, "whole exists:", whole.is_file())

index = PlacementIndex(work, score_id, whole)
print("aligned parts:", index.aligned_parts, "skipped:", index.skipped_parts)
print("slices:", len(index.slices))
non_empty = [(k, v) for k, v in index.slices.items() if any(v)]
print("slices containing any placement:", len(non_empty))
if non_empty:
    key, sl = non_empty[0]
    print("  example key:", key, "entries:", len(sl), "stated:", sum(1 for d in sl if d))
    page, system, part_index = key
    seg = work / "musicxml" / "unaligned" / f"{score_id}:{page:04d}:{system:04d}.musicxml"
    print("  segment exists:", seg.is_file())
    single = extract_part(ET.parse(seg).getroot(), part_index)
    applied = apply_placements(single.find("part"), sl)
    print("  applied:", applied)
    print("  placements now in the extracted part:",
          len([s for s in single.iter("slur") if s.get("placement")]))
