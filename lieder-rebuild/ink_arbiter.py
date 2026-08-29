"""Resolve the displacement direction with a metric independent of the alignment.

Every model-free check so far has been circular: the alignment is *built* to match the
barline counts detected in the crop, so the rebuilt labels agree with those counts by
construction. Ink is not. A crop showing dense music must carry a label with many
symbols, whatever the alignment believes, and the relationship can be calibrated on the
pairs where the two corpora AGREE - those are not in dispute.

Then for each disputed pair, ask which label the ink predicts. If the rebuilt labels are
displaced, their symbol counts will fit the ink worse than the old ones do, systematically.
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def note_count(tokens_path):
    n = 0
    for line in Path(tokens_path).read_text().splitlines():
        head = line.split()
        if head and (head[0].startswith("note") or head[0].startswith("rest")):
            n += 1
    return n

def ink(image_path):
    """Fraction of dark pixels, and dark pixels per unit width - both scale with how
    much music the crop holds, independent of anything the aligner decided."""
    img = Image.open(image_path).convert("L")
    a = np.asarray(img, dtype=np.uint8)
    if a.size == 0:
        return None
    dark = (a < 128)
    return {"frac": float(dark.mean()), "count": int(dark.sum()),
            "w": a.shape[1], "h": a.shape[0]}

disputed = json.loads(Path("/workspace/b0/lieder-rebuild/displaced_pairs.json").read_text())
disputed_stems = {d["stem"] for d in disputed}

# Calibration set: pairs both corpora label identically. Nothing disputed about these.
old_rows = {}
for line in Path("/workspace/b0/imslp_train_index.txt").read_text().splitlines():
    if line.strip():
        i, t = line.split(",", 1)
        old_rows[Path(i).stem] = t
new_rows = {}
for line in Path("/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt").read_text().splitlines():
    if line.strip():
        i, t = line.split(",", 1)
        new_rows[Path(i).stem] = (i, t)

def body(p):
    return tuple(l.strip() for l in Path(p).read_text().splitlines()
                 if l.strip() and not l.split()[0].startswith("timeSignatureBeats_"))

agree = []
for stem, (img, t) in new_rows.items():
    if stem in disputed_stems or stem not in old_rows:
        continue
    try:
        if body(t) == body(old_rows[stem]):
            agree.append((stem, img, t))
    except Exception:
        pass

print(f"calibration pairs (both corpora identical): {len(agree)}")
xs, ys = [], []
for stem, img, t in agree[:1200]:
    m = ink(img)
    if not m or m["count"] == 0:
        continue
    n = note_count(t)
    if n == 0:
        continue
    xs.append(m["count"]); ys.append(n)
xs, ys = np.array(xs, float), np.array(ys, float)
# Ink per symbol, robust to outliers.
ratio = np.median(xs / ys)
resid = np.abs(ys - xs / ratio) / np.maximum(ys, 1)
print(f"  ink pixels per symbol (median): {ratio:,.0f}")
print(f"  |predicted - actual| / actual : median {np.median(resid):.3f}, p90 {np.percentile(resid, 90):.3f}")

print(f"\ndisputed pairs: {len(disputed)}")
better_new = better_old = tie = skipped = 0
rows = []
for d in disputed:
    m = ink(d["image"])
    if not m or m["count"] == 0:
        skipped += 1
        continue
    pred = m["count"] / ratio
    n_new, n_old = note_count(d["new_tokens"]), note_count(d["old_tokens"])
    if n_new == n_old:
        tie += 1
        continue
    e_new, e_old = abs(pred - n_new), abs(pred - n_old)
    if e_new < e_old:
        better_new += 1
    else:
        better_old += 1
    rows.append({"stem": d["stem"], "offset": d["offset"], "ink_predicts": round(pred, 1),
                 "new": n_new, "old": n_old, "closer": "new" if e_new < e_old else "old"})

print(f"  ink favours the REBUILT label : {better_new}")
print(f"  ink favours the PRE-REBUILD   : {better_old}")
print(f"  identical symbol counts (uninformative): {tie}")
print(f"  skipped: {skipped}")
Path("/workspace/b0/lieder-rebuild/ink_arbiter.json").write_text(json.dumps(rows, indent=2))
print("\nexamples:")
for r in rows[:10]:
    print(f"  {r['stem']:>26} off {r['offset']:+d}  ink predicts {r['ink_predicts']:>6}  "
          f"new {r['new']:>3}  old {r['old']:>3}  -> {r['closer']}")
