"""Can the arithmetic repair fix the implicit-tuplet labels? (corrected)

The first attempt summed every note token in a bar, which counts each member of a chord
separately and inflates the total. Uses `measure_durations` here - the pipeline's own
function, which groups simultaneities first.

That function is only valid on SINGLE staves: on a grand staff `group_into_chords` takes
the minimum duration across a chord, so a bar where the hands differ is neither their sum
nor either hand's length. 89% of the quarantined pairs are grand staves, so the repair can
only be assessed on the remaining 11% - which is itself a finding about how far this can go.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path

from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords
from training.omr_datasets.audit_label_consistency import is_single_staff, measure_durations
from training.transformer.training_vocabulary import read_tokens

TUPLETS = ((3, 2), (6, 4), (5, 4), (7, 4), (9, 8))
PLAIN = {"8": "12", "16": "24", "4": "6", "32": "48", "2": "3"}


def dur(v):
    dotted = v.endswith(".")
    base = Fraction(1, int(v.rstrip(".")))
    return base * Fraction(3, 2) if dotted else base


def bar_values(syms):
    """Per-bar list of chord durations, matching how measure_durations totals them."""
    out, cur = [], []
    for chord in add_tuplet_start_stop(group_into_chords(syms)):
        if chord.is_barline():
            if cur:
                out.append(cur)
            cur = []
        else:
            d = chord.get_duration()
            if d > 0:
                cur.append(d)
    if cur:
        out.append(cur)
    return out


def analyse(syms):
    totals = measure_durations(syms)
    if len(totals) < 3:
        return 0, 0
    prevailing = Counter(totals).most_common(1)[0][0]
    bars = bar_values(syms)
    over = fixable = 0
    for i, total in enumerate(totals):
        if total <= prevailing * Fraction(21, 20):
            continue
        over += 1
        excess = total - prevailing
        vals = bars[i] if i < len(bars) else []
        hit = False
        for written, sounded in TUPLETS:
            if hit:
                break
            for plain in PLAIN:
                d = dur(plain)
                if excess != (written - sounded) * d:
                    continue
                run = [j for j, v in enumerate(vals) if v == d]
                for s in range(len(run) - written + 1):
                    w = run[s:s + written]
                    if w[-1] - w[0] == written - 1:
                        hit = True
                        break
                if hit:
                    break
        fixable += hit
    return over, fixable


man = Path("/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt")
stats = {"single": [0, 0, 0, 0], "grand": [0, 0, 0, 0]}
for line in man.read_text().splitlines():
    if not line.strip():
        continue
    syms = read_tokens(line.split(",", 1)[1])
    kind = "single" if is_single_staff(syms) else "grand"
    o, f = analyse(syms)
    if not o:
        continue
    s = stats[kind]
    s[0] += 1; s[1] += o; s[2] += f; s[3] += (f == o)

for kind, (pairs, over, fix, full) in stats.items():
    note = "" if kind == "single" else "   (durations NOT valid here - see docstring)"
    print(f"{kind} staves:{note}")
    print(f"  pairs with an overfull bar        : {pairs}")
    print(f"  overfull bars                     : {over}")
    print(f"  bars a tuplet rewrite makes EXACT : {fix}  ({100*fix/max(over,1):.1f}%)")
    print(f"  pairs fully repaired              : {full}")
