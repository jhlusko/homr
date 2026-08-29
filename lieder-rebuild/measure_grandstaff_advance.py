"""Validate advance_from_own_duration against real GrandStaff .krn files: no crashes,
sane distribution, and confirm the min-rule guarantee holds (no wild OTHER spikes that
would suggest the format assumption is wrong in practice)."""
import glob
import random
from collections import Counter
from pathlib import Path

from training.omr_datasets.humdrum_kern_parser import convert_kern_to_tokens

random.seed(7)
krn_files = glob.glob("/workspace/b0/homr/datasets/grandstaff/**/*.krn", recursive=True)
sample = random.sample(krn_files, min(200, len(krn_files)))

counts = Counter()
errors = 0
for path in sample:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        symbols = convert_kern_to_tokens(lines)
    except Exception:
        errors += 1
        continue
    for s in symbols:
        if s.notation is not None:
            counts[str(s.notation.advance)] += 1

print(f"sampled {len(sample)} of {len(krn_files)} .krn files, {errors} failed to parse")
total = sum(counts.values())
for cls, n in counts.most_common():
    print(f"  {cls:16s} {n:8,}  ({100*n/max(total,1):.2f}%)")
