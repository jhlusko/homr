"""Phase 0 measurement: real advance distribution on a sample of real Lieder scores."""
import glob
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens

random.seed(7)
mxl_files = glob.glob("/workspace/b0/homr/datasets/Lieder-main/scores/**/*.mxl", recursive=True)
sample = random.sample(mxl_files, min(60, len(mxl_files)))

advance_counts = Counter()
grand_staff_advance_counts = Counter()
total_note_symbols = 0
errors = 0
tmp = Path("/tmp/advance_sample.musicxml")

for mxl_path in sample:
    try:
        with zipfile.ZipFile(mxl_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".xml") and not n.startswith("META-INF")]
            if not names:
                continue
            tmp.write_bytes(zf.read(names[0]))
        measures = music_xml_file_to_tokens(str(tmp))
    except Exception:
        errors += 1
        continue
    for part in measures:
        for measure in part:
            for symbol in measure:
                if not symbol.rhythm.startswith(("note", "rest")):
                    continue
                total_note_symbols += 1
                if symbol.notation is None:
                    continue
                advance_counts[str(symbol.notation.advance)] += 1
                if symbol.position in ("upper", "lower"):
                    grand_staff_advance_counts[str(symbol.notation.advance)] += 1

print(f"sampled {len(sample)} scores, {errors} failed to parse")
print(f"total note/rest symbols: {total_note_symbols:,}")
print(f"\nadvance distribution (all):")
for cls, n in advance_counts.most_common():
    print(f"  {cls:16s} {n:8,}  ({100*n/max(sum(advance_counts.values()),1):.2f}%)")
print(f"\nadvance distribution (grand-staff symbols only, position=upper/lower):")
for cls, n in grand_staff_advance_counts.most_common():
    print(f"  {cls:16s} {n:8,}  ({100*n/max(sum(grand_staff_advance_counts.values()),1):.2f}%)")
