"""How much lyric supervision do the corpora we already have actually carry?

MusicXML gives lyrics semantically - <lyric> with <syllabic>, <text> and <extend> - so the
labels for a lyric stage would come from the same files the notation labels do. The
question is whether any corpus in hand has enough of them.
"""
import random
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


def read(path):
    if path.suffix == ".mxl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            inner = next(
                (n for n in names if n.endswith(".xml") and "/" not in n),
                next((n for n in names if n.endswith(".xml")), None),
            )
            if inner is None:
                return None
            return ET.fromstring(archive.read(inner).decode("utf-8"))
    return ET.parse(path).getroot()


def probe(name, paths, limit=150):
    counts = Counter()
    for path in paths[:limit]:
        try:
            root = read(path)
        except Exception:
            continue
        if root is None:
            continue
        counts["files"] += 1
        found = False
        for note in root.iter("note"):
            counts["notes"] += 1
            for lyric in note.findall("lyric"):
                found = True
                counts["lyrics"] += 1
                syllabic = lyric.findtext("syllabic")
                if syllabic:
                    counts["syllabic " + syllabic] += 1
                if lyric.find("extend") is not None:
                    counts["extend (melisma)"] += 1
                number = lyric.get("number")
                if number and number != "1":
                    counts["verse beyond the first"] += 1
        counts["with lyrics"] += found

    files = max(counts["files"], 1)
    print(f"{name}: {counts['files']} files, {counts['notes']:,} notes")
    share = counts["with lyrics"] / files
    print(f"   files carrying lyrics: {counts['with lyrics']} ({share:.0%})")
    if counts["lyrics"]:
        density = counts["lyrics"] / max(counts["notes"], 1)
        print(f"   lyric elements: {counts['lyrics']:,} ({density:.1%} of notes)")
        for key in sorted(counts):
            if key.startswith("syllabic ") or key in ("extend (melisma)", "verse beyond the first"):
                print(f"     {key}: {counts[key]:,}")
    print()


ossq = Path("/workspace/b0/ossq-omr")
probe("OSSQ (string quartets)", sorted(ossq.glob("scores/*/*/musicxml/unaligned/*.musicxml")))

olimpic = Path("/workspace/b0/olimpic-probe/olimpic-1.0-scanned/samples")
probe("OLiMPiC scanned (pianoform)", sorted(olimpic.glob("*/*.musicxml")))

pdmx = sorted(Path("/workspace/b0/homr/datasets/pdmx/mxl").glob("**/*.mxl"))
random.Random(0).shuffle(pdmx)
probe("PDMX", pdmx)
