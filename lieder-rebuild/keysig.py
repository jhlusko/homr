"""Does the model's reading of the key signature agree with the label's?

Asked before building anything. The scored .jsonl files already hold two independent
reads of the same staves - the label's key signature (from the transcription) and the
model's (from the scan) - in the rhythm branch, so the disagreement rate is a lookup,
not a project. If they already agree ~always, a dedicated detector has nothing to fix.
"""
import json, sys, collections
from pathlib import Path

def keysig(tokens):
    for t in tokens:
        if t.startswith("keySignature"):
            return t
    return None

for spec in sys.argv[1:]:
    label, _, path = spec.partition("=")
    both = agree = only_ref = only_pred = neither = 0
    confusion = collections.Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        r = keysig(row.get("rhythm_reference", []))
        p = keysig(row.get("rhythm_predicted", []))
        if r and p:
            both += 1
            if r == p: agree += 1
            else: confusion[(r, p)] += 1
        elif r: only_ref += 1
        elif p: only_pred += 1
        else: neither += 1
    n = both or 1
    print(f"{label}: {agree}/{both} agree = {100*agree/n:.1f}%  "
          f"(label-only {only_ref}, model-only {only_pred}, neither {neither})")
    for (r, p), c in confusion.most_common(6):
        print(f"    label {r:18s} model {p:18s} x{c}")
