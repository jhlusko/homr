"""Check the two corpus fixes in the built v6 data, not just in the build log's counts.

A log line saying "41 dropped" is the builder reporting on itself. These assertions read
the pairs back off disk.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
from homr.music_xml_generator import (
    add_tuplet_start_stop, group_into_chords,
    modal_measure_duration, stated_numerator_contradicts_bars,
)
from homr.transformer.vocabulary import TIME_SIGNATURE_BEATS_PREFIX
from training.omr_datasets.audit_label_consistency import is_single_staff, overfull_bars
from training.transformer.training_vocabulary import read_tokens

def stems(p):
    return {Path(l.split(",", 1)[0]).stem: Path(l.split(",", 1)[1])
            for l in Path(p).read_text().splitlines() if l.strip()}

v5 = stems("/workspace/b0/lieder-rebuild/stage2_clean_v5_manifest.txt")
v6 = stems("/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt")
over = stems("/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt")

restored = set(over) & set(v6)
print(f"FIX 1  overfull pairs restored into v6: {len(restored)} of {len(over)} quarantined")
grand = sum(1 for s in restored if not is_single_staff(read_tokens(str(v6[s]))))
print(f"       of those, grand staves: {grand}  (single staves: {len(restored) - grand})")

still_out = set(over) - set(v6)
single_out = sum(1 for s in still_out if s in v5 or True)
print(f"       still excluded: {len(still_out)}")
bad = [s for s in restored if is_single_staff(read_tokens(str(v6[s])))]
print(f"       ASSERT no single-staff overfull pair was restored: "
      f"{'PASS' if not bad else 'FAIL ' + str(bad[:3])}")

print()
contradicting = 0
carries = 0
for stem, path in v6.items():
    sym = read_tokens(str(path))
    if not any(s.rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX) for s in sym):
        continue
    carries += 1
    modal = modal_measure_duration(add_tuplet_start_stop(group_into_chords(sym)))
    if stated_numerator_contradicts_bars(sym, modal):
        contradicting += 1
print(f"FIX 2  v6 pairs still carrying a stated numerator: {carries}")
print(f"       of those, contradicting their own bars: {contradicting}")
print(f"       ASSERT none remain: {'PASS' if contradicting == 0 else 'FAIL'}")

print()
v5_contra = 0
for stem, path in v5.items():
    sym = read_tokens(str(path))
    if not any(s.rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX) for s in sym):
        continue
    modal = modal_measure_duration(add_tuplet_start_stop(group_into_chords(sym)))
    if stated_numerator_contradicts_bars(sym, modal):
        v5_contra += 1
print(f"       for comparison, v5 had {v5_contra} such pairs")
