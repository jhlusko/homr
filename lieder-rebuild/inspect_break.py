import json
from pathlib import Path
from homr.tuplet_repair import repair, split_bars, bar_duration, prevailing_bar, count_overfull
PAD = "\x00"
def real(s): return [t for t in s if not t.startswith(PAD)]

targets = {
    "/workspace/b0/phase7fix/valid/sq8075304_0012_0004_2.txt",
    "/workspace/b0/phase7fix/valid/sq8885571_0022_0005_1.txt",
}
for line in Path("/workspace/b0/lieder-rebuild/general_tuplet_s7.jsonl").read_text().splitlines():
    if not line.strip(): continue
    row = json.loads(line)
    if row.get("tokens") not in targets: continue
    want, got = real(row["rhythm_reference"]), real(row["rhythm_predicted"])
    fixed, rw = repair(got)
    bars = split_bars(got)
    prev = prevailing_bar(bars)
    print(f"\n=== {row['tokens']} ===")
    print(f"prevailing: {prev}  overfull count: {count_overfull(bars, prev)}")
    print(f"rewrites: {rw}")
    for i, (b, w) in enumerate(zip(bars, split_bars(want))):
        got_str = " ".join(b)
        want_str = " ".join(w)
        mark = "  <-- OVERFULL" if prev and bar_duration(b) > prev * 21 / 20 else ""
        diff = " DIFF" if got_str != want_str else ""
        if diff or mark:
            print(f" bar{i}: got=[{got_str}]  want=[{want_str}]{mark}{diff}")
