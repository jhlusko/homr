"""Repair implied tuplets in a prediction arithmetically, rather than learning them.

The model's largest error is writing three plain eighths where the page shows a triplet.
More training data barely helps: raising tuplet supply 70% cut tuplet-token errors ~6% and
produced no more correct staves. That is unsurprising - a tuplet is marked by a bracket a
few pixels wide, and the model cannot check its own arithmetic.

But the arithmetic is checkable. If a bar overruns the staff's prevailing bar by exactly
(n-m)*d, and it contains a run of n equal values of duration d, then rewriting that run as
an n-in-the-time-of-m tuplet makes the bar exact. Nothing else produces that signature.

This applies the repair to already-scored PREDICTIONS and measures what it recovers, so
the idea can be judged before any of it is wired into decoding.
"""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

PAD = "\x00"
DIV = {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
# (written, sounded) for the tuplets 19th-century engraving actually uses.
TUPLETS = ((3, 2), (6, 4), (5, 4), (7, 4), (9, 8))
PLAIN_TO_TUPLET = {"8": "12", "16": "24", "4": "6", "32": "48"}


def value(token):
    if not token.startswith("note_") and not token.startswith("rest_"):
        return None
    v = token.split("_", 1)[1]
    return v if v.rstrip(".").isdigit() else None


def duration(v):
    dotted = v.endswith(".")
    base = Fraction(1, int(v.rstrip(".")))
    return base * Fraction(3, 2) if dotted else base


def bars(tokens):
    out, cur = [], []
    for t in tokens:
        if t in DIV:
            out.append(cur); cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def repair(tokens):
    """Return (repaired tokens, number of runs rewritten)."""
    grouped = bars(tokens)
    lengths = []
    for b in grouped:
        total = Fraction(0)
        for t in b:
            v = value(t)
            if v:
                total += duration(v)
        lengths.append(total)
    real = [x for x in lengths if x > 0]
    if len(real) < 3:
        return tokens, 0
    prevailing = Counter(real).most_common(1)[0][0]
    fixed = 0
    out_bars = []
    for b, total in zip(grouped, lengths):
        excess = total - prevailing
        done = False
        if excess > 0:
            vals = [value(t) for t in b]
            for written, sounded in TUPLETS:
                if done:
                    break
                for plain, trip in PLAIN_TO_TUPLET.items():
                    d = duration(plain)
                    if excess != (written - sounded) * d:
                        continue
                    # Find a contiguous run of `written` notes of exactly this value.
                    run = [i for i, v in enumerate(vals) if v == plain]
                    for s in range(len(run) - written + 1):
                        window = run[s:s + written]
                        if window[-1] - window[0] != written - 1:
                            continue
                        b = list(b)
                        for i in window:
                            b[i] = b[i].replace(f"_{plain}", f"_{trip}")
                        fixed += 1
                        done = True
                        break
        out_bars.append(b)
    rebuilt = []
    divs = [t for t in tokens if t in DIV]
    for i, b in enumerate(out_bars):
        rebuilt.extend(b)
        if i < len(divs):
            rebuilt.append(divs[i])
    return rebuilt, fixed


path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/b0/lieder-rebuild/general_s7.jsonl"
before = after = total = staves = repaired_staves = runs = 0
for line in Path(path).read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    want = [t for t in row.get("rhythm_reference", []) if not t.startswith(PAD)]
    got = [t for t in row.get("rhythm_predicted", []) if not t.startswith(PAD)]
    if not want or not got:
        continue
    staves += 1
    fixedtok, n = repair(got)
    if n:
        repaired_staves += 1
        runs += n
    width = max(len(want), len(got))
    before += sum(1 for a, b in zip(want, got) if a == b)
    after += sum(1 for a, b in zip(want, fixedtok) if a == b)
    total += width
print(f"{Path(path).name}: {staves} staves")
print(f"  staves where the repair fired: {repaired_staves}  ({100*repaired_staves/max(staves,1):.1f}%)")
print(f"  runs rewritten as tuplets    : {runs}")
print(f"  rhythm-token matches  before : {before:,}")
print(f"                         after : {after:,}   ({after-before:+,})")
