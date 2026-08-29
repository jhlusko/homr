"""Where on the benchmark does Lieder fine-tuning help, and where does it hurt?

The corpus profile says OSSQ sits between our two training sources: Lieder single
staves run a median of 16 symbols over 3-4 bars, PDMX replay 72 symbols over 8, and
OSSQ 25 symbols over ~6. If that mismatch is what limits transfer, the effect should be
visible as a gradient - fine-tuning helping where the benchmark looks like Lieder and
hurting where it does not.

Uses only already-scored predictions. No training, no GPU.
"""
import json
import sys
from pathlib import Path

R = "/workspace/b0/lieder-rebuild"
RUNS = {"426 base": f"{R}/general_old.jsonl",
        "447": f"{R}/general_mid.jsonl",
        "456 v6 s7": f"{R}/general_s7.jsonl"}

def load(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        correct = total = 0
        for branch in ("pitch", "rhythm", "lift", "articulation", "slur", "position"):
            want, got = row.get(f"{branch}_reference"), row.get(f"{branch}_predicted")
            if want is None:
                continue
            correct += sum(1 for a, b in zip(want, got) if a == b)
            total += max(len(want), len(got))
        # Length of the reference is the staff's density: how many symbols it holds.
        ref = row.get("rhythm_reference") or []
        n = sum(1 for t in ref if not t.startswith("\x00"))
        out[row["tokens"]] = (correct, total, n)
    return out

runs = {name: load(p) for name, p in RUNS.items()}
shared = set.intersection(*(set(r) for r in runs.values()))
print(f"{len(shared)} staves scored by all {len(runs)} runs\n")

# Buckets by symbol count, chosen so each holds a usable number of staves.
EDGES = [(0, 15), (15, 22), (22, 30), (30, 45), (45, 10**9)]
print(f"{'symbols':>12} {'staves':>7} " + " ".join(f"{n:>10}" for n in runs) + "   447-426   456-426")
for lo, hi in EDGES:
    ids = [i for i in shared if lo <= runs["426 base"][i][2] < hi]
    if not ids:
        continue
    acc = {}
    for name, r in runs.items():
        c = sum(r[i][0] for i in ids); t = sum(r[i][1] for i in ids)
        acc[name] = 100 * c / t if t else float("nan")
    label = f"{lo}-{hi if hi < 10**9 else '+'}"
    print(f"{label:>12} {len(ids):>7} " + " ".join(f"{acc[n]:>10.2f}" for n in runs)
          + f"   {acc['447'] - acc['426 base']:>+7.2f}   {acc['456 v6 s7'] - acc['426 base']:>+7.2f}")
