"""Did training on naturals-restored Lieder regress anything on PDMX (which has none)?"""
import json, sys
from pathlib import Path
PAD = "\x00"
BRANCHES = ("rhythm", "pitch", "lift", "articulation", "slur", "position")
def real(s): return [t for t in s if not t.startswith(PAD)]

for path in sys.argv[1:]:
    tot = {b: 0 for b in BRANCHES}
    hit = {b: 0 for b in BRANCHES}
    exact = staves = 0
    n_pred = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        want_r, got_r = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
        if not want_r or not got_r: continue
        staves += 1
        all_ok = True
        for b in BRANCHES:
            want, got = real(row.get(f"{b}_reference", [])), real(row.get(f"{b}_predicted", []))
            w = min(len(want), len(got))
            for i in range(w):
                tot[b] += 1
                ok = want[i] == got[i]
                hit[b] += ok
                if not ok: all_ok = False
            if len(want) != len(got): all_ok = False
            if b == "lift":
                n_pred += sum(1 for t in got if t == "N")
        exact += all_ok
    print(f"\n{Path(path).name}: staves={staves}  exact={exact} ({100*exact/max(staves,1):.2f}%)  N_predicted={n_pred}")
    for b in BRANCHES:
        print(f"  {b:14s} {100*hit[b]/max(tot[b],1):.2f}%")
