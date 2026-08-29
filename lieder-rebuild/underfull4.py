"""Interior UNDERFULL bars after removing the three benign explanations.

An interior bar shorter than the staff's own modal bar is only evidence of dropped
content if none of these apply:
  * whole-measure rest  - the label writes `rest_1` for the bar-rest glyph whatever
    the metre, so every bar-rest in 3/4 or 9/8 reads short by construction;
  * a section change    - a doublebarline or repeat inside the crop, after which a
    pickup or a different metre is expected;
  * a stated metre change - a `timeSignature/` token appearing after the first bar.
Each exclusion is counted so the residual is visible against what it was carved from.
"""
import sys, json, collections
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import simultaneity_groups

UNDER = Fraction(19, 20)
HARD = ("doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatEndStart",
        "voltaStart", "voltaStop", "voltaDiscontinue")
BAR = ("barline",) + HARD

def staff_bars(symbols):
    durs = {"upper": [Fraction(0)], "lower": [Fraction(0)]}
    kinds = {"upper": [[]], "lower": [[]]}
    hard = [False]
    metre_change = [False]
    seen_lower = False
    for g in simultaneity_groups(symbols):
        bl = [s.rhythm for s in g if s.rhythm in BAR]
        if bl:
            hard.append(any(b in HARD for b in bl))
            metre_change.append(False)
            for p in durs:
                durs[p].append(Fraction(0)); kinds[p].append([])
            continue
        if any(s.rhythm.startswith(("timeSignature/", "timeSignatureBeats_")) for s in g):
            metre_change[-1] = True
        for p in ("upper", "lower"):
            mem = [s for s in g if s.position == p and s.rhythm.startswith(("note_", "rest_"))]
            if not mem:
                continue
            if p == "lower":
                seen_lower = True
            durs[p][-1] += max(s.get_duration().fraction for s in mem)
            kinds[p][-1].extend(s.rhythm for s in mem)
    if not seen_lower:
        durs.pop("lower"); kinds.pop("lower")
    return durs, kinds, hard, metre_change

rows = [l.split(",", 1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
st = collections.Counter()
residual = []
shortfall = collections.Counter()
for image, tokens in rows:
    st["pairs"] += 1
    try:
        sym = read_tokens(tokens)
    except Exception:
        st["unreadable"] += 1; continue
    if any(s.rhythm.startswith("rest_") and s.rhythm.endswith("m") for s in sym):
        st["multirest_pairs_skipped"] += 1; continue
    durs, kinds, hard, mchange = staff_bars(sym)
    grand = "lower" in durs
    tag = "grand" if grand else "single"
    # a section change or a stated metre change anywhere invalidates "prevailing"
    any_hard = any(hard[1:])
    any_metre = any(mchange[1:])
    for pos, seq in durs.items():
        kk = kinds[pos]
        while seq and seq[-1] == 0:
            seq.pop(); kk.pop()
        if len(seq) < 4:
            st[f"staves_too_few_bars_{tag}"] += 1; continue
        st[f"staves_{tag}"] += 1
        st[f"interior_bars_{tag}"] += len(seq) - 2
        prevailing = collections.Counter(seq).most_common(1)[0][0]
        if prevailing <= 0:
            continue
        for i in range(1, len(seq) - 1):
            if not (0 < seq[i] < prevailing * UNDER):
                continue
            st[f"raw_underfull_{tag}"] += 1
            if kk[i] and all(k in ("rest_1", "rest_0") for k in kk[i]):
                st[f"excl_whole_measure_rest_{tag}"] += 1; continue
            if any_hard:
                st[f"excl_section_change_{tag}"] += 1; continue
            if any_metre:
                st[f"excl_metre_change_{tag}"] += 1; continue
            st[f"RESIDUAL_{tag}"] += 1
            shortfall[str(prevailing - seq[i])] += 1
            if len(residual) < 60:
                residual.append({"stem": Path(image).stem, "pos": pos, "bar": i,
                                 "prevailing": str(prevailing),
                                 "bars": [str(d) for d in seq], "content": kk[i]})
print(json.dumps({"stats": dict(st), "shortfall": shortfall.most_common(12),
                  "residual": residual}, indent=2, default=str))
