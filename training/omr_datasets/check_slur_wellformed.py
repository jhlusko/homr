"""Are the slur markers in the corpus well-formed?

A slur is a FIELD on a note token, not a token of its own, so a broken slur cannot change
a symbol count - it is not a candidate for the length errors that dominate the benchmark.
It is still a label defect: the model is trained to predict that field, and it is being
shown impossible configurations.

Two classes are legitimate and must not be counted as defects:

* a stop with no start near the BEGINNING of a crop - the slur began on the previous
  system and was correctly carried in;
* a start with no stop near the END - it continues onto the next system.

What cannot be explained that way, on a MONOPHONIC single staff where a stack model is
exact:

* a stop with no open slur, well past the start of the stream;
* a start while one is already open - on a single voice there is nothing to nest with.

**Most of what this finds is NOT a corpus defect.** The slur vocabulary holds exactly
five values - `nonote`, `empty`, `slurStart`, `slurStop`, `slurStart_slurStop` - one
field per note, with no way to record how many slurs are open. Nesting is therefore
unrepresentable, and a phrase slur drawn over inner slurs is standard engraving in
Lieder. A "nested start" is the format failing to express what the page correctly shows.

That also explains the ~96 Verovio warnings across 106 rendered scores: the generator
cannot pair slurs reliably because the token stream does not carry the pairing. They are
a symptom of a representation limit, not of bad labels, and no corpus fix will remove
them - only a vocabulary carrying slur depth or identity would.

Corpus-wide the counts are close to balanced: 9,134 starts against 9,436 stops, a 3%
excess of stops consistent with crops cut at system boundaries.

Run this to localise the cases, not to count defects.
"""
import re
from collections import Counter
from pathlib import Path

EDGE = 0.15  # a marker inside this fraction of either end is a plausible system carry


def analyse(tokens_path):
    notes = []
    for line in Path(tokens_path).read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("note"):
            continue
        slur = parts[4] if len(parts) > 4 else "_"
        pos = parts[5] if len(parts) > 5 else "_"
        notes.append((slur, pos))
    if not notes:
        return None
    # Monophonic only: one staff position, no chords implied by repeated positions.
    if len({p for _, p in notes if p != "_"}) > 1:
        return None
    n = len(notes)
    depth = 0
    unmatched_stop_late = 0
    nested_start = 0
    open_at_end = 0
    for index, (slur, _) in enumerate(notes):
        frac = index / max(n - 1, 1)
        if "slurStart" in slur:
            if depth > 0:
                nested_start += 1
            depth += 1
        if "slurStop" in slur:
            if depth == 0:
                if frac > EDGE:
                    unmatched_stop_late += 1
            else:
                depth -= 1
    open_at_end = depth
    return {"notes": n, "unmatched_stop_late": unmatched_stop_late,
            "nested_start": nested_start, "open_at_end": open_at_end}


manifest = Path("/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt")
rows = [l.split(",", 1) for l in manifest.read_text().splitlines() if l.strip()]
mono = 0
defects = Counter()
affected = set()
per_score = Counter()
for image, tokens in rows:
    r = analyse(tokens)
    if r is None:
        continue
    mono += 1
    bad = r["unmatched_stop_late"] + r["nested_start"]
    if bad:
        affected.add(image)
        defects["unmatched_stop_late"] += r["unmatched_stop_late"]
        defects["nested_start"] += r["nested_start"]
        m = re.search(r"(IMSLP\d+)-sys", image)
        if m:
            per_score[m.group(1)] += bad

print(f"{len(rows):,} pairs, {mono:,} monophonic single-staff (where the stack model is exact)")
print(f"pairs with an unexplainable slur marker: {len(affected):,} "
      f"({100*len(affected)/max(mono,1):.1f}% of monophonic)")
for k, v in defects.most_common():
    print(f"   {k:>22}: {v}")
print(f"scores affected: {len(per_score)}")
print("worst scores:", ", ".join(f"{s}({n})" for s, n in per_score.most_common(5)))
