"""Are the corpus's unmatched slur endpoints the expected system-break kind?

A crop cut out of a system legitimately begins inside a slur (an unmatched slurStop
near the START) and ends inside one (an unmatched slurStart near the END).  Anything
else - an unmatched stop in the middle, a start opened while one is already open -
is a label defect the model is trained on.  Position is measured as the fraction of
the note stream, per staff position, so the two cases separate.
"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens

rows = [l.split(",", 1) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
st = collections.Counter()
stop_pos = collections.Counter()
start_pos = collections.Counter()
mid_examples = []
for image, tokens in rows:
    st["pairs"] += 1
    try:
        sym = read_tokens(tokens)
    except Exception:
        st["unreadable"] += 1; continue
    if any(x.position == "lower" for x in sym):
        st["skip_grand"] += 1; continue
    if any(x.rhythm == "chord" for x in sym):
        st["skip_chorded"] += 1; continue
    st["monophonic_single_staff"] += 1
    streams = collections.defaultdict(list)
    for s in sym:
        if s.rhythm.startswith("note_"):
            streams[s.position].append(s)
    bad = False
    for pos, notes in streams.items():
        n = len(notes)
        if n == 0:
            continue
        depth = 0
        for i, s in enumerate(notes):
            tags = s.slur.split("_")
            frac = i / max(n - 1, 1)
            if "slurStop" in tags:
                if depth == 0:
                    st["unmatched_stop"] += 1
                    stop_pos[min(int(frac * 10), 9)] += 1
                    if frac > 0.15:
                        st["unmatched_stop_midstream"] += 1; bad = True
                        if len(mid_examples) < 10:
                            mid_examples.append({"stem": Path(image).stem, "pos": pos,
                                                 "note_index": i, "of": n})
                else:
                    depth -= 1
                    st["matched_slur"] += 1
            if "slurStart" in tags:
                if depth > 0:
                    st["nested_start"] += 1; bad = True
                depth += 1
        if depth:
            st["unmatched_start"] += depth
            # where did the last open start sit?
            for i, s in enumerate(notes):
                pass
            opens = [i for i, s in enumerate(notes) if "slurStart" in s.slur.split("_")]
            if opens:
                start_pos[min(int(opens[-1] / max(n - 1, 1) * 10), 9)] += 1
    if bad:
        st["pairs_with_malformed_slur"] += 1
print(json.dumps({"stats": dict(st),
                  "unmatched_stop_decile": [stop_pos[i] for i in range(10)],
                  "last_open_start_decile": [start_pos[i] for i in range(10)],
                  "midstream_examples": mid_examples}, indent=2))
