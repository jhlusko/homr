"""Is the missing barline the LAST one?

Predictions with a bar-count mismatch are mostly one divider short while being longer
overall, so the model is not truncating - it is over-generating notes and omitting a
divider. If the omitted one is the system-final barline, that is both a specific defect
and a cheap one to address, since almost every scanned system ends on a barline.
"""
import json
from collections import Counter
from pathlib import Path

PAD = "\x00"
DIV = {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}

def real(seq):
    return [t for t in seq if not t.startswith(PAD)]

R = "/workspace/b0/lieder-rebuild"
for label, path in (("426 base", f"{R}/general_old.jsonl"), ("456 v6", f"{R}/general_s7.jsonl")):
    ends_ref = ends_got = 0
    n = 0
    missing_final = Counter()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ref, got = real(row.get("rhythm_reference", [])), real(row.get("rhythm_predicted", []))
        if not ref or not got:
            continue
        n += 1
        r_end = ref[-1] in DIV
        g_end = got[-1] in DIV
        ends_ref += r_end
        ends_got += g_end
        b_ref = sum(1 for t in ref if t in DIV)
        b_got = sum(1 for t in got if t in DIV)
        if b_got - b_ref == -1:
            missing_final["reference ends on a divider, prediction does not" if (r_end and not g_end)
                          else "the missing divider is elsewhere"] += 1
    print(f"\n=== {label}: {n} staves ===")
    print(f"  references ending on a divider : {ends_ref}  ({100*ends_ref/n:.1f}%)")
    print(f"  predictions ending on a divider: {ends_got}  ({100*ends_got/n:.1f}%)")
    print(f"  among the one-bar-short staves:")
    for k, v in missing_final.most_common():
        print(f"     {k}: {v}")
