"""How much of the overfull discard is grand-staff duration arithmetic that is known
to be invalid?

audit_label_consistency.py skips grand staves for every duration-dependent check, and
says why: the token format attributes a simultaneity's duration ONCE, so a chord
spanning both staves gives its duration to one and zero to the other. A grand staff is
one rhythmic stream, not two. build_clean_stage2_pairs calls overfull_bars() with no
such guard.
"""
import sys, collections, json
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import is_single_staff
from training.omr_datasets.make_tuplet_review_set import classify

rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
by = collections.Counter()
bars = collections.Counter()
for image, tokens in rows:
    try: sym = read_tokens(tokens)
    except Exception: continue
    single = is_single_staff(sym)
    f = classify(sym)
    if not f: continue
    by[("single" if single else "grand", )] += 1
    for x in f:
        bars[("single" if single else "grand", x["kind"])] += 1
print("PAIRS   ", {k[0]: v for k, v in by.items()})
print("\nBARS by staff type and cause")
for (st, k), v in sorted(bars.items()):
    print(f"  {st:7s} {k:18s} {v:4d}")
