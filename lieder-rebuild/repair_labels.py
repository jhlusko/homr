"""Can the arithmetic repair fix the IMPLICIT-tuplet labels themselves?

An implicit tuplet carries no bracket and no numeral. Nothing in the image says it is one,
so the model cannot learn to read it - and the transcriber did not mark it either, which
is exactly why the bar sums overfull. The label is wrong in the same way the prediction is.

That reframes the 417 pairs the builder quarantines. They are not bad data to discard;
they are data whose labels need repairing. Restoring them raw made tuplet errors worse
(324 -> 352) because the plain values teach wrong durations. Restoring them REPAIRED may
be a different proposition.

Measures how many overfull bars the repair makes exact, on the labels rather than on
predictions.
"""
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from training.transformer.training_vocabulary import read_tokens

DIV = {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
TUPLETS = ((3, 2), (6, 4), (5, 4), (7, 4), (9, 8))
PLAIN_TO_TUPLET = {"8": "12", "16": "24", "4": "6", "32": "48", "2": "3"}


def dur(v):
    dotted = v.endswith(".")
    base = Fraction(1, int(v.rstrip(".")))
    return base * Fraction(3, 2) if dotted else base


def note_val(sym):
    if not sym.rhythm.startswith(("note_", "rest_")):
        return None
    v = sym.rhythm.split("_", 1)[1]
    return v if v.rstrip(".").isdigit() else None


def bar_split(syms):
    out, cur = [], []
    for s in syms:
        if s.rhythm in DIV:
            out.append(cur); cur = []
        else:
            cur.append(s)
    if cur:
        out.append(cur)
    return out


def analyse(syms):
    """(#overfull bars, #of those a tuplet rewrite makes exact)."""
    grouped = bar_split(syms)
    lengths = []
    for b in grouped:
        t = Fraction(0)
        for s in b:
            v = note_val(s)
            if v:
                t += dur(v)
        lengths.append(t)
    real = [x for x in lengths if x > 0]
    if len(real) < 3:
        return 0, 0
    prevailing = Counter(real).most_common(1)[0][0]
    over = fixable = 0
    for b, total in zip(grouped, lengths):
        if total <= prevailing * Fraction(21, 20):
            continue
        over += 1
        excess = total - prevailing
        vals = [note_val(s) for s in b]
        hit = False
        for written, sounded in TUPLETS:
            if hit:
                break
            for plain in PLAIN_TO_TUPLET:
                if excess != (written - sounded) * dur(plain):
                    continue
                run = [i for i, v in enumerate(vals) if v == plain]
                for s0 in range(len(run) - written + 1):
                    w = run[s0:s0 + written]
                    if w[-1] - w[0] == written - 1:
                        hit = True
                        break
                if hit:
                    break
        fixable += hit
    return over, fixable


man = Path("/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt")
pairs = over_total = fix_total = pairs_fully = 0
for line in man.read_text().splitlines():
    if not line.strip():
        continue
    syms = read_tokens(line.split(",", 1)[1])
    o, f = analyse(syms)
    if not o:
        continue
    pairs += 1
    over_total += o
    fix_total += f
    pairs_fully += (f == o)
print(f"quarantined pairs with an overfull bar: {pairs}")
print(f"  overfull bars total                 : {over_total}")
print(f"  bars a tuplet rewrite makes EXACT   : {fix_total}  ({100*fix_total/max(over_total,1):.1f}%)")
print(f"  pairs where EVERY overfull bar fixes: {pairs_fully}  ({100*pairs_fully/max(pairs,1):.1f}%)")
