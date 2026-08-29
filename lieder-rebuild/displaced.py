"""List the stems whose rebuilt label equals a NEIGHBOURING system's old label.

The displacement is real as an observation - two corpora disagree, and 100 of 364
disagreements are exactly an off-by-N system - but nothing has established which side is
displaced. No structural check finds a mechanism, and every model-free arbiter available
is circular: the alignment is *built* to match the detected barline counts, so of course
the new labels match them.

The crop is the only non-circular arbiter, which makes this a review question. This
writes the manifest for it.
"""
import json
import re
from pathlib import Path

def pairs(p):
    d = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            i, t = line.split(",", 1)
            d[Path(i).stem] = (Path(i), Path(t))
    return d

def body(p):
    return tuple(l.strip() for l in p.read_text().splitlines()
                 if l.strip() and not l.split()[0].startswith("timeSignatureBeats_"))

old = pairs("/workspace/b0/imslp_train_index.txt")
new = pairs("/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt")

oldidx = {}
for stem, (_, t) in old.items():
    m = re.match(r"^(.+)-sys(\d+)-v(\d+)$", stem)
    if m:
        oldidx.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = (t, stem)

found = []
for stem, (img, t) in new.items():
    m = re.match(r"^(.+)-sys(\d+)-v(\d+)$", stem)
    if not m:
        continue
    score, sysno, voice = m.group(1), int(m.group(2)), m.group(3)
    table = oldidx.get((score, voice))
    if not table or sysno not in table:
        continue
    nb = body(t)
    if nb == body(table[sysno][0]):
        continue
    for off in (-3, -2, -1, 1, 2, 3):
        nb2 = table.get(sysno + off)
        if nb2 and body(nb2[0]) == nb:
            found.append({"stem": stem, "offset": off,
                          "image": str(img),
                          "new_tokens": str(t),
                          "old_tokens": str(table[sysno][0]),
                          "old_stem_matched": nb2[1]})
            break

print(f"{len(found)} stems in the rebuilt corpus carry a neighbouring system's old label")
from collections import Counter
print("offsets:", dict(Counter(f["offset"] for f in found).most_common()))
print("scores affected:", len({f["stem"].split("-sys")[0] for f in found}))
out = Path("/workspace/b0/lieder-rebuild/displaced_pairs.json")
out.write_text(json.dumps(found, indent=2))
print(f"wrote {out}")
# A manifest the review generator can consume: crop, new label, old label.
man = Path("/workspace/b0/lieder-rebuild/displaced_manifest.txt")
man.write_text("\n".join(f"{f['image']},{f['new_tokens']}" for f in found) + "\n")
oldman = Path("/workspace/b0/lieder-rebuild/displaced_old_manifest.txt")
oldman.write_text("\n".join(f"{f['image']},{f['old_tokens']}" for f in found) + "\n")
print(f"wrote {man} and {oldman}")
