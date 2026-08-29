"""Classify the overfull bars by cause, rather than assuming all are implied tuplets."""
import json, sys, collections
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import measure_durations

def bar_chords(symbols):
    bars, cur = [], []
    for chord in add_tuplet_start_stop(group_into_chords(symbols)):
        if chord.is_barline():
            if cur: bars.append(cur)
            cur = []
        elif chord.get_duration() > 0:
            cur.append(chord.get_duration())
    if cur: bars.append(cur)
    return bars

CANDIDATES = [(3,2,"triplet"), (6,4,"sextuplet"), (5,4,"quintuplet"),
              (7,4,"septuplet"), (9,8,"nonuplet")]

def tuplet_explains(durations, excess):
    counts = collections.Counter(durations)
    for n, m, name in CANDIDATES:
        for d, have in counts.items():
            if have >= n and excess == (n - m) * d:
                return name
    return None

rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
cls = collections.Counter()
pair_cls = collections.Counter()
buckets = collections.defaultdict(list)
for image, tokens in rows:
    try: sym = read_tokens(tokens)
    except Exception: continue
    durs = measure_durations(sym)
    if len(durs) < 3: continue
    counts = collections.Counter(durs)
    prevailing = counts.most_common(1)[0][0]
    bars = bar_chords(sym)
    bad = [i for i, d in enumerate(durs) if d > prevailing * Fraction(21,20)]
    kinds = set()
    for i in bad:
        ratio = Fraction(durs[i], prevailing)
        excess = durs[i] - prevailing
        if ratio.denominator == 1 and ratio >= 2:
            k = "integer_multiple(missing barline?)"
        elif tuplet_explains(bars[i] if i < len(bars) else [], excess):
            k = "tuplet_" + tuplet_explains(bars[i] if i < len(bars) else [], excess)
        # A crop that genuinely changes metre has no single prevailing bar: the
        # second most common length is well represented and the "overfull" bars
        # simply belong to the other metre.
        elif len(counts) > 1 and counts.most_common(2)[1][1] >= 2 and durs[i] == counts.most_common(2)[1][0]:
            k = "second_metre(metre change?)"
        else:
            k = "unexplained"
        cls[k] += 1
        kinds.add(k)
    if kinds:
        pair_cls["+".join(sorted(kinds)) if len(kinds) > 1 else next(iter(kinds))] += 1
        buckets[next(iter(kinds)) if len(kinds)==1 else "mixed"].append(Path(image).stem)

print("BARS BY CAUSE"); [print(f"  {k:38s} {v:4d}") for k,v in cls.most_common()]
print("\nPAIRS BY CAUSE"); [print(f"  {k:60s} {v:4d}") for k,v in pair_cls.most_common()]
Path(sys.argv[2]).write_text(json.dumps({k: v for k, v in buckets.items()}, indent=2))
