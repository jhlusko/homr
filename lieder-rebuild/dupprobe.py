"""Unison duplicates: how many note tokens are a second copy of a note already in the
same simultaneity on the same staff, and how many are byte-identical to it.

A byte-identical pair is one notehead in the engraving carrying two tokens in the
label - the model is asked to emit a symbol the page does not show.  Same pitch with
a different value is two voices in unison, engraved as one head with two stems, which
is the same problem in weaker form.
"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import simultaneity_groups

rows = [l.split(",", 1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
st = collections.Counter()
ex = collections.defaultdict(list)
for image, tokens in rows:
    try:
        sym = read_tokens(tokens)
    except Exception:
        continue
    st["pairs"] += 1
    grand = any(s.position == "lower" for s in sym)
    tag = "grand" if grand else "single"
    st[f"pairs_{tag}"] += 1
    notes = [s for s in sym if s.rhythm.startswith("note_")]
    st[f"notes_{tag}"] += len(notes)
    ident = same_pitch = 0
    for g in simultaneity_groups(sym):
        for pos in ("upper", "lower"):
            mem = [s for s in g if s.position == pos and s.rhythm.startswith("note_")]
            if len(mem) < 2:
                continue
            keys = [(s.rhythm, s.pitch, s.lift) for s in mem]
            pitches = [(s.pitch, s.lift) for s in mem]
            ident += len(keys) - len(set(keys))
            same_pitch += len(pitches) - len(set(pitches))
    if ident:
        st[f"pairs_with_identical_{tag}"] += 1
        st[f"identical_extra_notes_{tag}"] += ident
        if len(ex[tag]) < 12:
            ex[tag].append({"stem": Path(image).stem, "extra": ident, "notes": len(notes)})
    st[f"same_pitch_extra_notes_{tag}"] += same_pitch
    if same_pitch:
        st[f"pairs_with_same_pitch_{tag}"] += 1
print(json.dumps({"stats": dict(st), "examples": dict(ex)}, indent=2, default=str))
