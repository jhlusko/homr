"""How much tuplet error is ADDRESSABLE by arithmetic, and what still breaks?"""
import json, sys
from fractions import Fraction
from pathlib import Path
from homr.tuplet_repair import (repair, split_bars, bar_duration, prevailing_bar,
                                count_overfull, PLAIN_TO_TUPLET, OVERFULL_RATIO)
PAD = "\x00"
TUP = frozenset(PLAIN_TO_TUPLET.values())
def real(s): return [t for t in s if not t.startswith(PAD)]
def ntup(seq): return sum(1 for t in seq if t.startswith(("note_","rest_"))
                          and t.split("_",1)[1].rstrip(".") in TUP)

path = sys.argv[1]
missed = under = no_overfull = multi_overfull = no_candidate = 0
broken = []
for line in Path(path).read_text().splitlines():
    if not line.strip(): continue
    row = json.loads(line)
    want, got = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
    if not want or not got: continue
    fixed, rw = repair(got)
    if rw and got == want and fixed != want:
        broken.append(row.get("tokens", "?"))
    r, p = ntup(want), ntup(got)
    if r > p:                      # reference has tuplets the model did not write
        missed += 1
        bars = split_bars(got)
        prev = prevailing_bar(bars)
        if prev is None:
            under += 1
        elif count_overfull(bars, prev) == 0:
            no_overfull += 1
        elif count_overfull(bars, prev) > 1:
            multi_overfull += 1
        elif not rw:
            no_candidate += 1

print(f"=== {Path(path).name} ===")
print(f"staves where the model wrote FEWER tuplets than the reference: {missed}")
print(f"  of those, the prediction is NOT overfull anywhere      : {no_overfull}")
print(f"  more than one overfull bar (guard declines)            : {multi_overfull}")
print(f"  one overfull bar but no exact tuplet rewrite fits      : {no_candidate}")
print(f"  too few bars to establish a prevailing length          : {under}")
print(f"staves the repair still BREAKS: {broken}")
