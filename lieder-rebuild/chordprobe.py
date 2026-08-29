"""Malformed simultaneities: a `chord` separator is a claim that two symbols sound
together, and some claims are impossible.

  duplicate_pitch      the same pitch+octave twice in one simultaneity on one staff -
                       one symbol more than the engraving can show
  identical_symbol     the same full symbol repeated in one simultaneity
  chord_with_structural  a barline, clef or signature joined into a simultaneity
  chord_of_one         a `chord` separator with nothing on one side
  unequal_durations    members of one staff's simultaneity holding different values
"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
from training.omr_datasets.audit_label_consistency import simultaneity_groups

STRUCT = ("barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd",
          "repeatEndStart", "voltaStart", "voltaStop", "voltaDiscontinue")
rows = [l.split(",", 1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
st = collections.Counter()
ex = collections.defaultdict(list)
for image, tokens in rows:
    st["pairs"] += 1
    try:
        sym = read_tokens(tokens)
    except Exception:
        st["unreadable"] += 1; continue
    raw = Path(tokens).read_text().splitlines()
    for line in raw:
        parts = [p for p in line.split("&")]
        if any(not p.strip() for p in parts):
            st["empty_chord_member"] += 1
            ex["empty_chord_member"].append(Path(image).stem)
    grand = any(s.position == "lower" for s in sym)
    tag = "grand" if grand else "single"
    hit = set()
    for g in simultaneity_groups(sym):
        st[f"simultaneities_{tag}"] += 1
        if len(g) > 1:
            st[f"chords_{tag}"] += 1
        else:
            continue
        if any(s.rhythm in STRUCT or s.rhythm.startswith(("clef_", "keySignature_", "timeSignature")) for s in g):
            if not all(s.rhythm in STRUCT or s.rhythm.startswith(("clef_", "keySignature_", "timeSignature")) for s in g):
                st[f"chord_with_structural_{tag}"] += 1; hit.add("struct")
                if len(ex["struct"]) < 6: ex["struct"].append((Path(image).stem, [str(s) for s in g]))
            continue
        for pos in ("upper", "lower"):
            mem = [s for s in g if s.position == pos and s.rhythm.startswith("note_")]
            pitches = [s.pitch for s in mem]
            if len(pitches) != len(set(pitches)):
                st[f"duplicate_pitch_{tag}"] += 1; hit.add("dup")
                if len(ex["dup"]) < 8: ex["dup"].append((Path(image).stem, [str(s) for s in g]))
            full = [str(s) for s in mem]
            if len(full) != len(set(full)):
                st[f"identical_symbol_{tag}"] += 1; hit.add("ident")
            durs = {s.rhythm for s in g if s.position == pos and s.rhythm.startswith(("note_", "rest_"))}
            if len(durs) > 1:
                st[f"unequal_durations_{tag}"] += 1; hit.add("uneq")
        if any(s.rhythm.startswith("rest_") for s in g) and any(s.rhythm.startswith("note_") for s in g):
            poss = {s.position for s in g}
            if len(poss) == 1:
                st[f"rest_with_note_same_staff_{tag}"] += 1; hit.add("restnote")
    for h in hit:
        st[f"pairs_with_{h}"] += 1
print(json.dumps({"stats": dict(st), "examples": {k: v[:6] for k, v in ex.items()}}, indent=2, default=str))
