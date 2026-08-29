"""Review set for the displacement: same crop, rebuilt label against the old one.

126 stems in the current corpus carry a label identical to a NEIGHBOURING system's label
in the pre-rebuild corpus, offsets skewing negative 98 to 28. One of the two corpora is
displaced and no structural check can say which: the alignment is built to match the
detected barline counts, so every model-free arbiter agrees with the rebuild by
construction. The crop is the only independent evidence, which makes this a review
question rather than an analysis one.

Left is the rebuilt label, right is the old one, and the scan decides.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
from training.omr_datasets.make_review_sets import build_set

found = json.loads(Path("/workspace/b0/lieder-rebuild/displaced_pairs.json").read_text())
left = {f["stem"]: f"{f['image']},{f['new_tokens']}" for f in found}
right = {f["stem"]: f"{f['image']},{f['old_tokens']}" for f in found}
extra = {f["stem"]: {"kind": "displaced", "offset": f["offset"],
                     "matches_old_system": f["old_stem_matched"]} for f in found}

summary = build_set("displaced", sorted(left), left, right,
                    Path("/workspace/b0/lieder-rebuild/review_displaced"), 126, extra)
print(json.dumps(summary, indent=2))
