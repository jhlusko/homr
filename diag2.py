"""Find a part phase2 converted and phase4 refused, and show why."""
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.convert_ossq import extract_part
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.transformer.training_vocabulary import to_decoder_branches

def stems(index):
    return {Path(l.split(",")[1]).stem for l in Path(index).read_text().splitlines() if l.strip()}

before = stems("/workspace/b0/phase2/train/index.txt")
after = stems("/workspace/b0/phase4/train/index.txt")
lost = sorted(before - after)
print(f"phase2 {len(before):,}  phase4 {len(after):,}  newly refused {len(lost):,}")
print("examples:", lost[:3])

score, page, system, part = lost[0].rsplit("_", 3)
seg = next(Path("/workspace/b0/ossq-omr").glob(f"scores/*/*/musicxml/unaligned/{score}:{page}:{system}.musicxml"))
print("segment:", seg.name, "part", part)

tree = ET.parse(seg).getroot()
single = extract_part(tree, int(part) - 1)
scratch = Path("/tmp/diag2.musicxml")
scratch.write_text('<?xml version="1.0"?>\n' + ET.tostring(single, encoding="unicode"))
syms = [s for v in music_xml_file_to_tokens(str(scratch)) for m in v for s in m]
try:
    to_decoder_branches(syms)
    print("loads fine now (?)")
except KeyError as e:
    print("REFUSED:", e)
    for s in syms:
        if s.slur and s.slur not in ("", "slurStart", "slurStop", "slurStart_slurStop"):
            print("  offending symbol slur field:", repr(s.slur), "rhythm", s.rhythm)
            break

# what does the source note look like?
for note in single.iter("note"):
    n = note.find("notations")
    if n is not None and len(n.findall("slur")) + len(n.findall("tied")) >= 2:
        print("source note:", ET.tostring(n, encoding="unicode").strip()[:220])
        break
