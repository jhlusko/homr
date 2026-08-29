"""Write a >=45-symbol subset of each scored jsonl, restricted to staves present in all
of them, so compare_checkpoints reports the dense cut instead of the 4pp-noisy aggregate."""
import json, sys, pathlib
PAD = "\x00"
paths = [pathlib.Path(p) for p in sys.argv[1:]]
rows = []
for p in paths:
    d = {}
    for line in p.open():
        r = json.loads(line)
        d[r["tokens"]] = r
    rows.append(d)
common = set(rows[0])
for d in rows[1:]:
    common &= set(d)
dense = {k for k in common
         if len([t for t in rows[0][k]["rhythm_reference"] if not t.startswith(PAD)]) >= 45}
print(f"{len(common)} common staves, {len(dense)} dense", file=sys.stderr)
for p, d in zip(paths, rows):
    out = p.with_name(p.stem + "_dense.jsonl")
    with out.open("w") as f:
        for k in sorted(dense):
            f.write(json.dumps(d[k]) + "\n")
    print(out)
