"""Confirm real advance targets landed in the rebuilt corpus, and get its raw scale."""
from collections import Counter
from pathlib import Path
from training.transformer.training_vocabulary import read_tokens

man = Path("/workspace/b0/lieder-rebuild/stage2_clean_advance_probe_manifest.txt")
lines = [l for l in man.read_text().splitlines() if l.strip()]
counts = Counter()
grand_pairs = single_pairs = 0
for line in lines:
    tokens_path = line.split(",", 1)[1]
    symbols = read_tokens(tokens_path)
    from training.omr_datasets.notation_sidecar import attach_sidecar
    attach_sidecar(tokens_path, symbols)
    if any(s.position == "lower" for s in symbols):
        grand_pairs += 1
    else:
        single_pairs += 1
    for s in symbols:
        if s.notation is not None:
            counts[str(s.notation.advance)] += 1

print(f"pairs: {len(lines)}  grand-staff: {grand_pairs}  single-staff: {single_pairs}")
total = sum(counts.values())
for cls, n in counts.most_common():
    print(f"  {cls:16s} {n:8,}  ({100*n/max(total,1):.2f}%)")
