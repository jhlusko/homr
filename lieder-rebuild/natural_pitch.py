"""Does stripping a natural change the SOUNDING pitch of the rendered score?

A natural cancels an accidental the key signature would otherwise apply. If the label
records `N` the renderer emits that natural; if `strip_naturals` has turned it into
`empty`, the renderer has nothing to stop the key signature applying, and an F-natural in
G major should come back out as F#.

If that happens, stripping is not a labelling convention - it is a correctness bug that
alters the music. Compares the MusicXML generated from the same pairs with and without
stripping.
"""
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from homr.circle_of_fifths import strip_naturals
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.transformer.training_vocabulary import read_tokens

manifest = Path("/workspace/b0/lieder-rebuild/stage2_clean_naturals_manifest.txt")
if not manifest.exists():
    sys.exit("naturals-kept manifest not built")

def sounding(symbols):
    xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
    out = []
    for note in xml.iter("note"):
        p = note.find("pitch")
        if p is None:
            continue
        step = p.findtext("step") or "?"
        alter = p.findtext("alter") or "0"
        octv = p.findtext("octave") or "?"
        out.append(f"{step}{alter}/{octv}")
    return out

rows = [l.split(",", 1) for l in manifest.read_text().splitlines() if l.strip()]
checked = differing = 0
examples = []
diffs = Counter()
for image, tokens in rows:
    syms = read_tokens(tokens)
    if not any(s.lift == "N" for s in syms):
        continue
    checked += 1
    try:
        kept = sounding(syms)
        stripped = sounding(strip_naturals(syms))
    except Exception:
        continue
    if kept != stripped:
        differing += 1
        for a, b in zip(kept, stripped):
            if a != b:
                diffs[f"{a} -> {b}"] += 1
        if len(examples) < 5:
            bad = [(a, b) for a, b in zip(kept, stripped) if a != b][:3]
            examples.append((Path(image).stem, bad))
    if checked >= 400:
        break

print(f"pairs containing a natural, checked: {checked}")
print(f"pairs whose SOUNDING pitches change when stripped: {differing} "
      f"({100*differing/max(checked,1):.1f}%)")
print("\nmost common pitch changes (kept -> stripped), step+alter/octave:")
for k, v in diffs.most_common(8):
    print(f"   {k}   {v}")
print("\nexamples:")
for stem, bad in examples:
    print(f"   {stem}: {bad}")
