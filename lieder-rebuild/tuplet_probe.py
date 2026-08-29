"""How much of the overfull-bar loss a tuplet hypothesis would explain."""
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
        else:
            if chord.get_duration() > 0: cur.append(chord.get_duration())
    if cur: bars.append(cur)
    return bars

# A tuplet of N notes in the time of M writes N*d where M*d sounds, so the bar is
# long by exactly (N-M)*d. Try the tuplets 19th-century engraving actually leaves
# unmarked, smallest first, and require a real run of N equal values to carry it.
CANDIDATES = [(3,2,"triplet"), (6,4,"sextuplet"), (5,4,"quintuplet"),
              (7,4,"septuplet"), (9,8,"nonuplet"), (2,3,"duplet")]

def explain(durations, excess):
    counts = collections.Counter(durations)
    for n, m, name in CANDIDATES:
        for d, have in counts.items():
            if have >= n and excess == (n - m) * d:
                return name, d
    return None, None

rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
stats = collections.Counter()
ratios = collections.Counter()
named = collections.Counter()
detail = []
for image, tokens in rows:
    try:
        sym = read_tokens(tokens)
    except Exception:
        stats["unreadable"] += 1; continue
    durs = measure_durations(sym)
    if len(durs) < 3:
        stats["too_short"] += 1; continue
    prevailing = collections.Counter(durs).most_common(1)[0][0]
    bars = bar_chords(sym)
    bad = [i for i, d in enumerate(durs) if d > prevailing * Fraction(21,20)]
    stats["pairs"] += 1
    stats["bad_bars"] += len(bad)
    per_pair_ok = True
    for i in bad:
        excess = durs[i] - prevailing
        ratios[str(Fraction(durs[i], prevailing))] += 1
        name, d = explain(bars[i] if i < len(bars) else [], excess)
        if name:
            named[name] += 1; stats["explained_bars"] += 1
        else:
            per_pair_ok = False
            named["unexplained"] += 1
    if per_pair_ok and bad:
        stats["fully_explained_pairs"] += 1
        detail.append(Path(image).stem)

print(json.dumps({"stats": dict(stats),
                  "bar_ratio_actual_over_prevailing": dict(ratios.most_common(15)),
                  "tuplet_hypothesis": dict(named.most_common())}, indent=2))
Path(sys.argv[2]).write_text("\n".join(detail) + "\n")
print(f"\nfully explained pairs listed in {sys.argv[2]}: {len(detail)}")
