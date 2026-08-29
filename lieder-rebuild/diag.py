"""What separates the staves the repair fixes from the ones it breaks?"""
import json, sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from homr.tuplet_repair import repair, split_bars, bar_duration, prevailing_bar

PAD = "\x00"
def real(s): return [t for t in s if not t.startswith(PAD)]

def profile(tokens):
    bars = split_bars(tokens)
    lens = [d for d in map(bar_duration, bars) if d > 0]
    p = prevailing_bar(bars)
    if p is None or not lens:
        return None
    return {"bars": len(lens), "prevailing": p,
            "modal_share": sum(1 for d in lens if d == p) / len(lens),
            "overfull": sum(1 for d in lens if d > p * Fraction(21, 20))}

for path in sys.argv[1:]:
    print(f"\n=== {Path(path).name} ===")
    groups = {"recovered": [], "broken": [], "other": []}
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        want, got = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
        if not want or not got: continue
        fixed, rw = repair(got)
        if not rw: continue
        was, now = got == want, fixed == want
        k = "recovered" if (now and not was) else "broken" if (was and not now) else "other"
        pr = profile(got)
        if pr: groups[k].append(pr)
    for k, rows in groups.items():
        if not rows: continue
        share = sorted(r["modal_share"] for r in rows)
        over = Counter(r["overfull"] for r in rows)
        prev = Counter(str(r["prevailing"]) for r in rows)
        print(f"  {k:10s} n={len(rows):3d}  modal share min/med/max = "
              f"{share[0]:.2f}/{share[len(share)//2]:.2f}/{share[-1]:.2f}  "
              f"overfull bars {dict(over.most_common(4))}")
        print(f"             prevailing: {dict(prev.most_common(5))}")
