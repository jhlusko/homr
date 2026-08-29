"""Is the sparse-staff regression real, or 23 staves of noise?

Stratification showed both Lieder fine-tunes losing ~1.8pp on staves under 15 symbols
while gaining on dense ones. That bucket holds 23 staves, small enough that the sign
could be chance - so this checks it three ways: against a wider sparse cut, per-branch,
and against the two same-corpus seeds, whose difference is pure noise and therefore
calibrates what a 23-staff bucket can show.
"""
import json
from pathlib import Path

R = "/workspace/b0/lieder-rebuild"
RUNS = {"426": f"{R}/general_old.jsonl", "447": f"{R}/general_mid.jsonl",
        "455 s42": f"{R}/general_v6fix.jsonl", "456 s7": f"{R}/general_s7.jsonl"}
BRANCHES = ("pitch", "rhythm", "lift", "articulation", "slur", "position")

def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        per = {}
        for br in BRANCHES:
            want, got = row.get(f"{br}_reference"), row.get(f"{br}_predicted")
            if want is None:
                continue
            per[br] = (sum(1 for a, b in zip(want, got) if a == b), max(len(want), len(got)))
        n = sum(1 for t in (row.get("rhythm_reference") or []) if not t.startswith("\x00"))
        out[row["tokens"]] = (per, n)
    return out

runs = {k: load(v) for k, v in RUNS.items()}
shared = sorted(set.intersection(*(set(r) for r in runs.values())))

def acc(name, ids, branch=None):
    c = t = 0
    for i in ids:
        per = runs[name][0 if False else i][0]
        for br, (cc, tt) in per.items():
            if branch and br != branch:
                continue
            c += cc; t += tt
    return 100 * c / t if t else float("nan")

for cut in (15, 20, 25, 30, 45, 200):
    ids = [i for i in shared if runs["426"][i][1] >= cut if cut in (45,) else runs["426"][i][1] < cut]
    print(f"\nstaves under {cut} symbols: {len(ids)}")
    base = acc("426", ids)
    for n in ("447", "455 s42", "456 s7"):
        print(f"   {n:>9}  {acc(n, ids):6.2f}   vs 426 {acc(n, ids) - base:+6.2f}")
    print(f"   {'seed noise':>9}  same corpus, two seeds: {acc('456 s7', ids) - acc('455 s42', ids):+6.2f}")

ids = [i for i in shared if runs["426"][i][1] < 15]
print(f"\nper branch on the {len(ids)} sparsest staves, 447 vs 426")
for br in BRANCHES:
    print(f"   {br:>13}  {acc('447', ids, br) - acc('426', ids, br):+6.2f}")
