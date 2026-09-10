"""Why does a part that converted before now produce slurStart_slurStart?

The token slur field is built by _collect_articulation, which appends "slur"+type for
<slur> AND for <tied>. A note carrying both therefore yields two identical entries.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.convert_ossq import extract_part
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.omr_datasets.slur_placement import PlacementIndex, apply_placements
from training.transformer.training_vocabulary import to_decoder_branches

root = Path("/workspace/b0/ossq-omr")
# find a segment with a note carrying both <tied> and <slur>
for seg in sorted(root.glob("scores/*/*/musicxml/unaligned/*.musicxml"))[:400]:
    tree = ET.parse(seg).getroot()
    for pi, part in enumerate(tree.findall("part")):
        for note in part.iter("note"):
            n = note.find("notations")
            if n is not None and n.findall("tied") and n.findall("slur"):
                print("segment:", seg.name, "part", pi)
                single = extract_part(tree, pi)
                scratch = Path("/tmp/diag.musicxml")
                scratch.write_text('<?xml version="1.0"?>\n' + ET.tostring(single, encoding="unicode"))
                voices = music_xml_file_to_tokens(str(scratch))
                syms = [s for v in voices for m in v for s in m]
                bad = [s for s in syms if "_" in (s.slur or "")]
                print("  symbols with a compound slur field:", len(bad))
                for s in bad[:3]:
                    print("   ", repr(s.slur))
                try:
                    to_decoder_branches(syms)
                    print("  loads fine")
                except KeyError as e:
                    print("  REFUSED:", e)
                raise SystemExit
