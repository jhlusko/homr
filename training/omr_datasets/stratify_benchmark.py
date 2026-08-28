"""Where on the benchmark does Lieder fine-tuning help, and where does it hurt?

The corpus profile says OSSQ sits between our two training sources: Lieder single
staves run a median of 16 symbols over 3-4 bars, PDMX replay 72 symbols over 8, and
OSSQ 25 symbols over ~6. If that mismatch is what limits transfer, the effect should be
visible as a gradient - fine-tuning helping where the benchmark looks like Lieder and
hurting where it does not.

Uses only already-scored predictions. No training, no GPU.

**The first result refuted the hypothesis it was written to test.** Fine-tuning on
Lieder hurts the sparsest staves and helps the densest - the ones least like the
training data - so the density mismatch does not explain the transfer pattern.
"""
import json
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="a scored .jsonl; the first is the baseline others are shown against")
args = parser.parse_args()
RUNS = dict(spec.split("=", 1) for spec in args.run)

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
base = next(iter(runs))
others = [n for n in runs if n != base]
print(f"{'symbols':>12} {'staves':>7} " + " ".join(f"{n:>12}" for n in runs)
      + "".join(f"   {n} - {base}" for n in others))
for lo, hi in EDGES:
    ids = [i for i in shared if lo <= runs["426 base"][i][2] < hi]
    if not ids:
        continue
    acc = {}
    for name, r in runs.items():
        c = sum(r[i][0] for i in ids); t = sum(r[i][1] for i in ids)
        acc[name] = 100 * c / t if t else float("nan")
    label = f"{lo}-{hi if hi < 10**9 else '+'}"
    print(f"{label:>12} {len(ids):>7} " + " ".join(f"{acc[n]:>12.2f}" for n in runs)
          + "".join(f"   {acc[n] - acc[base]:>+8.2f}" for n in others))
