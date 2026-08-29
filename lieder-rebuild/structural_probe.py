"""What do the structural failures look like - truncation, or something else?

Structural errors (the prediction carrying a different number of measure dividers from
the reference) are the one failure no corpus change has moved: 5 of 148 dense staves at
baseline and 5-6 after every fine-tune, and the corpus's own bar grids are perfectly
self-consistent. So the cause is in the model or the decode.

The obvious decode-side hypothesis is truncation: generate() runs to max_seq_len=608 and
stops, or emits EOS early, and the trailing bars are simply never produced. That predicts
predictions SHORTER than references, and predictions clustering at the cap.
"""
import json
from collections import Counter
from pathlib import Path

PAD = "\x00"
DIV = {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
MAX_SEQ_LEN = 608

def real(seq):
    return [t for t in seq if not t.startswith(PAD)]

R = "/workspace/b0/lieder-rebuild"
for label, path in (("426 base", f"{R}/general_old.jsonl"), ("456 v6", f"{R}/general_s7.jsonl")):
    short = longer = 0
    at_cap = 0
    deltas = Counter()
    lens = []
    total_struct = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref, got = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
        b_ref = sum(1 for t in ref if t in DIV)
        b_got = sum(1 for t in got if t in DIV)
        if b_ref == b_got:
            continue
        total_struct += 1
        deltas[b_got - b_ref] += 1
        lens.append(len(got))
        if len(got) >= MAX_SEQ_LEN - 8:
            at_cap += 1
        if len(got) < len(ref):
            short += 1
        else:
            longer += 1
    print(f"\n=== {label}: {total_struct} staves with a bar-count mismatch ===")
    print(f"  prediction shorter than reference: {short}   longer: {longer}")
    print(f"  predictions at/near max_seq_len ({MAX_SEQ_LEN}): {at_cap}")
    if lens:
        lens.sort()
        print(f"  predicted length: min {lens[0]} median {lens[len(lens)//2]} max {lens[-1]}")
    print(f"  bar-count delta (predicted - reference): {dict(sorted(deltas.items()))}")
