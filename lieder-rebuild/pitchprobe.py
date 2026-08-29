"""Pitch plausibility: notes far off the staff for their clef, and implausible leaps.

Staff position is measured in diatonic steps from the clef's own anchor (the middle
line), so 4 steps is the top/bottom line and anything past 8 needs three ledger lines.
A misread octave or a wrong clef in the label shows up as a run of such notes or as a
leap of an octave or more between neighbours.
"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens, NOTE_TO_DIATONIC, CLEF_ANCHORS

def dia(p):
    return NOTE_TO_DIATONIC[p[0]] + 7 * int(p[1:])

rows = [l.split(",",1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
st = collections.Counter(); ex = []
hist = collections.Counter()
for image, tokens in rows:
    try: sym = read_tokens(tokens)
    except Exception: continue
    st["pairs"] += 1
    clef = {"upper": None, "lower": None}
    prev = {"upper": None, "lower": None}
    worst = 0
    for s in sym:
        if s.rhythm.startswith("clef_"):
            if s.rhythm in CLEF_ANCHORS:
                clef[s.position if s.position in clef else "upper"] = dia(CLEF_ANCHORS[s.rhythm])
            continue
        if not s.rhythm.startswith("note_"): continue
        pos = s.position if s.position in clef else "upper"
        st["notes"] += 1
        a = clef[pos]
        if a is not None:
            off = abs(dia(s.pitch) - a)
            hist[min(off, 20)] += 1
            if off > 10:
                st["beyond_4_ledger"] += 1
                worst = max(worst, off)
        p = prev[pos]
        if p is not None:
            leap = abs(dia(s.pitch) - p)
            if leap >= 14: st["leap_ge_2_octaves"] += 1
            elif leap >= 11: st["leap_ge_11_steps"] += 1
        prev[pos] = dia(s.pitch)
    if worst:
        st["pairs_with_extreme"] += 1
        if len(ex) < 12: ex.append({"stem": Path(image).stem, "max_offset_steps": worst})
print(json.dumps({"stats": dict(st), "offset_hist": [hist[i] for i in range(21)],
                  "examples": ex}, indent=2))
