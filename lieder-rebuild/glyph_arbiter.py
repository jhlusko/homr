"""Resolve the displacement using glyphs that are visible in the crop.

Note counts cannot arbitrate - the ink calibration has a 31% median error, wider than
the difference between the candidate labels. Structural glyphs can: a mid-system clef
change or a time signature is a large, unambiguous mark, and either the crop shows it or
it does not.

Visual inspection of IMSLP624193-sys6-v1 found the crop carrying a bass-clef change and
a 3/8 time signature that ONLY the rebuilt label contains. This asks how general that is.

A label claiming a glyph the crop lacks is as wrong as one missing a glyph the crop has,
so both directions are counted.
"""
import json
from collections import Counter
from pathlib import Path

STRUCTURAL = ("clef_", "timeSignature")

def glyphs(path):
    out = Counter()
    for line in Path(path).read_text().splitlines():
        head = line.split()
        if head and head[0].startswith(STRUCTURAL):
            out[head[0]] += 1
    return out

disputed = json.loads(Path("/workspace/b0/lieder-rebuild/displaced_pairs.json").read_text())
new_only = old_only = same = 0
detail = []
for d in disputed:
    gn, go = glyphs(d["new_tokens"]), glyphs(d["old_tokens"])
    only_new = gn - go
    only_old = go - gn
    if not only_new and not only_old:
        same += 1
        continue
    if only_new and not only_old:
        new_only += 1
        verdict = "rebuilt claims glyphs the old lacks"
    elif only_old and not only_new:
        old_only += 1
        verdict = "old claims glyphs the rebuilt lacks"
    else:
        verdict = "both claim glyphs the other lacks"
    detail.append({"stem": d["stem"], "offset": d["offset"], "verdict": verdict,
                   "only_in_rebuilt": dict(only_new), "only_in_old": dict(only_old)})

both = len([d for d in detail if d["verdict"].startswith("both")])
print(f"disputed pairs: {len(disputed)}")
print(f"  identical structural glyphs (uninformative): {same}")
print(f"  ONLY the rebuilt label has extra glyphs:      {new_only}")
print(f"  ONLY the old label has extra glyphs:          {old_only}")
print(f"  each has glyphs the other lacks:              {both}")
print("\nwhich glyphs the rebuilt label adds:")
add = Counter()
for d in detail:
    for k, v in d["only_in_rebuilt"].items():
        add[k] += v
for k, v in add.most_common(8):
    print(f"    {k:>22}  {v}")
print("\nwhich glyphs the OLD label adds:")
sub = Counter()
for d in detail:
    for k, v in d["only_in_old"].items():
        sub[k] += v
for k, v in sub.most_common(8):
    print(f"    {k:>22}  {v}")
Path("/workspace/b0/lieder-rebuild/glyph_arbiter.json").write_text(json.dumps(detail, indent=2))
