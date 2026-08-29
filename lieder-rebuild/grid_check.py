"""Do the voices of a system agree on how many bars it has?

Structural errors - the model losing the bar grid - are the one failure no corpus change
has moved: 5 of 148 dense staves at baseline and 5-6 after every fine-tune. If the
corpus itself carries inconsistent grids, the model is being taught them.

This is the rare check that is NOT circular. Build-time validation compares a label's
divider count against its aligned span, and the span comes from barlines detected in the
image, so it agrees with the detector by construction. Cross-voice agreement uses no
detection at all: a system's staves are the same bars of the same music, so if voice 0
says four bars and voice 1 says five, one of them is wrong whatever the detector thinks.
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

DIVIDERS = {"barline", "doublebarline", "bolddoublebarline",
            "repeatStart", "repeatEnd", "repeatBoth"}

def dividers(path):
    n = 0
    for line in Path(path).read_text().splitlines():
        head = line.split()
        if head and head[0] in DIVIDERS:
            n += 1
    return n

for name, manifest in (("v6 clean", "/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt"),
                       ("v5 clean", "/workspace/b0/lieder-rebuild/stage2_clean_v5_manifest.txt"),
                       ("reverse v3", "/workspace/b0/lieder-rebuild/stage2_reverse_manifest_v3.txt")):
    p = Path(manifest)
    if not p.exists():
        print(f"{name}: manifest missing")
        continue
    systems = defaultdict(dict)
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        img, tok = line.split(",", 1)
        m = re.match(r"^(.+)-sys(\d+)-v(\d+)$", Path(img).stem)
        if m:
            systems[(m.group(1), int(m.group(2)))][int(m.group(3))] = tok
    multi = {k: v for k, v in systems.items() if len(v) > 1}
    bad = []
    for key, voices in multi.items():
        counts = {v: dividers(t) for v, t in voices.items()}
        if len(set(counts.values())) > 1:
            bad.append((key, counts))
    print(f"\n{name}: {len(systems)} systems, {len(multi)} with more than one voice")
    print(f"  voices disagree on bar count: {len(bad)}  ({100*len(bad)/max(len(multi),1):.1f}% of multi-voice systems)")
    if bad:
        spread = Counter(max(c.values()) - min(c.values()) for _, c in bad)
        print(f"  size of disagreement: {dict(sorted(spread.items()))}")
        for key, counts in bad[:5]:
            print(f"    {key[0]}-sys{key[1]}: {counts}")
