"""How much is the end-on-a-divider invariant worth, if decoding enforced it?

Every reference in the benchmark ends on a measure divider - 792 of 792 - because a
scanned system is cut at a barline. Predictions violate that 23 times for 456 and 6
times for the baseline.

This simulates the constraint post hoc: append the divider the reference ends on to any
prediction that lacks one, and rescore. It is an upper bound on what a decode-time
constraint could recover, not the constraint itself - the model would also have to stop
at the right place - but it sizes the prize before any code is written.
"""
import json
from pathlib import Path

PAD = "\x00"
DIV = {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
BRANCHES = ("pitch", "rhythm", "lift", "articulation", "slur", "position")

def real(seq):
    return [t for t in seq if not t.startswith(PAD)]

def score(rows, fix):
    c = t = 0
    struct = 0
    for row in rows:
        per = {}
        for br in BRANCHES:
            w = real(row.get(f"{br}_reference", []))
            g = real(row.get(f"{br}_predicted", []))
            per[br] = (w, g)
        ref_r, got_r = per["rhythm"]
        if fix and ref_r and got_r and ref_r[-1] in DIV and got_r[-1] not in DIV:
            # Append the divider, and a matching filler on every other branch so the
            # streams stay aligned the way the model would have emitted them.
            for br in BRANCHES:
                w, g = per[br]
                g = g + [ref_r[-1] if br == "rhythm" else w[-1] if w else "_"]
                per[br] = (w, g)
        b_ref = sum(1 for x in per["rhythm"][0] if x in DIV)
        b_got = sum(1 for x in per["rhythm"][1] if x in DIV)
        if b_ref != b_got:
            struct += 1
        for br in BRANCHES:
            w, g = per[br]
            c += sum(1 for a, b in zip(w, g) if a == b)
            t += max(len(w), len(g))
    return 100 * c / t, struct

R = "/workspace/b0/lieder-rebuild"
for label, path in (("426 base", f"{R}/general_old.jsonl"), ("456 v6", f"{R}/general_s7.jsonl"),
                    ("447", f"{R}/general_mid.jsonl")):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    dense = [r for r in rows if len(real(r.get("rhythm_reference", []))) >= 45]
    for name, subset in (("all 792", rows), ("dense 45+", dense)):
        a0, s0 = score(subset, False)
        a1, s1 = score(subset, True)
        print(f"{label:>9} {name:>10}: {a0:6.2f} -> {a1:6.2f}  ({a1-a0:+.2f})   "
              f"bar-count mismatches {s0} -> {s1}")
    print()
