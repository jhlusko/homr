"""
Manufacture Fingering and SystemText training examples, by rendering real notation
that just happens not to occur often - not fabricated pixels.

27.89 fixed the detector's within-page positive sampling and found it could not move
Fingering or SystemText off 0% precision: both classes occur so rarely in the source
corpus (12 and 3 boxes across 2,847 pages, 27.68) that no amount of resampling existing
examples manufactures a signal that was never there. `musescore_boxes.py`'s own render
pipeline is the way out - it draws whatever the MusicXML says to draw, so a `<fingering>`
element added to a note, or a `<direction>` added to a measure, is rendered by the exact
same MuseScore binary and turned into a box by the exact same SVG-class extraction as
every other page in the corpus (27.25's rule: boxes from the renderer that drew the
image). This is synthetic in the sense that the annotation was not in the original score,
not in the sense of a faked image - the pixels are real MuseScore engraving of real
MusicXML content.

**Fingering is unambiguous, SystemText is a guess to be verified before it is trusted.**
A `<technical><fingering>` on a note is always a fingering; MuseScore does not reclassify
it. A `<direction><words>`, by contrast, is turned into Tempo, StaffText, Expression or
SystemText by a MuseScore-internal rule the MusicXML does not spell out
(`musescore_boxes.py`'s own docstring on `VERIFIABLE_CLASSES`). `system="only-top"` is the
one MusicXML attribute that plausibly maps to "printed once above the top staff, not per
staff" - which is what SystemText means in MuseScore's own engraving - but this is a
hypothesis, not a documented mapping, and `verify_svg_classes` exists to check it against
a real render rather than assume it.
"""

# flake8: noqa: T201

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.omr_datasets.musescore_boxes import Unrenderable, annotate

#: Plausible finger numbers. MuseScore renders whatever string is given; single digits
#: are what real fingering notation actually uses.
FINGER_NUMBERS = ("1", "2", "3", "4", "5")

#: Short, plainly textual annotations. First-pass testing on a real render found
#: `system="only-top"` alone is not enough - "Coda" and "D.C." rendered as Tempo
#: regardless (4 SystemText paths + 2 Tempo paths for 2 words, confirmed with
#: `verify_svg_classes`/grep against the SVG), so navigation/tempo-adjacent words are
#: excluded here rather than trusted on the `system` attribute alone.
SYSTEM_TEXT_WORDS = ("Chorus", "Refrain", "Ped.")


def _notes(root: ET.Element) -> list[ET.Element]:
    """Every real (non-rest) note across every part, in document order."""
    found = []
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                if note.find("rest") is None and note.find("pitch") is not None:
                    found.append(note)
    return found


def _measures(root: ET.Element) -> list[ET.Element]:
    """Every measure across every part, excluding each part's first.

    A `<direction><words>` placed in a score's opening measure rendered as Tempo in
    testing regardless of the `system="only-top"` attribute - MuseScore's own text-style
    default for the very start of a piece, where a tempo marking conventionally sits,
    independent of what this module asks for. Excluding the first measure of each part
    sidesteps that default rather than fighting it.
    """
    found = []
    for part in root.findall("part"):
        found.extend(part.findall("measure")[1:])
    return found


def inject_fingering(root: ET.Element, count: int, rng: random.Random) -> int:
    """Add a `<fingering>` to `count` random notes; returns how many were actually added
    (fewer than requested only if the score has fewer eligible notes than `count`)."""
    candidates = _notes(root)
    rng.shuffle(candidates)
    added = 0
    for note in candidates[:count]:
        notations = ET.Element("notations")
        technical = ET.SubElement(notations, "technical")
        fingering = ET.SubElement(technical, "fingering")
        fingering.text = rng.choice(FINGER_NUMBERS)
        # MusicXML's declared order puts <notations> before <lyric> - a lyric-bearing
        # note (this corpus is Lieder) would fail MuseScore's strict schema validation if
        # notations were simply appended after an existing lyric.
        lyric_index = next(
            (i for i, child in enumerate(note) if child.tag == "lyric"), len(note)
        )
        note.insert(lyric_index, notations)
        added += 1
    return added


def inject_system_text(root: ET.Element, count: int, rng: random.Random) -> int:
    """Add a `<direction>` word annotation, `system="only-top"`, to `count` random
    measures; returns how many were actually added."""
    candidates = _measures(root)
    rng.shuffle(candidates)
    added = 0
    for measure in candidates[:count]:
        direction = ET.Element("direction", {"placement": "above", "system": "only-top"})
        direction_type = ET.SubElement(direction, "direction-type")
        words = ET.SubElement(direction_type, "words")
        words.text = rng.choice(SYSTEM_TEXT_WORDS)
        # Directions are sequenced content, same as notes/attributes/barline - inserting
        # at the front of the measure is always schema-valid, unlike appending after a
        # note (which MusicXML's declared element order would reject).
        measure.insert(0, direction)
        added += 1
    return added


def augment(source: Path, target: Path, fingering: int, system_text: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    tree = ET.parse(source)
    root = tree.getroot()
    counts = {
        "fingering": inject_fingering(root, fingering, rng),
        "system_text": inject_system_text(root, system_text, rng),
    }
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return counts


def verify_svg_classes(svg_path: Path) -> Counter:
    """What MuseScore actually drew, by SVG class - the check the module's own docstring
    says the `system="only-top"` hypothesis needs before it is trusted."""
    text = svg_path.read_text(encoding="utf-8")
    return Counter(cls for cls in ("Fingering", "SystemText", "StaffText", "Expression", "Tempo") if f'class="{cls}"' in text)


def build_batch(
    sources: list[Path],
    out_root: Path,
    fingering: int,
    system_text: int,
    seed: int,
) -> dict:
    """Augment and render each source score, writing `{stem}_aug/` under `out_root` in
    the same shape `musescore_boxes.annotate` writes for the rest of the corpus - so
    these folders can be pointed at `detector_masks.py` exactly like `/workspace/b0/mbox`.

    Kept to training data only is the caller's job: this writes wherever it is told, and
    a caller wanting an honest before/after box-eval comparison (as 27.89's retrain did)
    must not point the *validation* split's index at anything this function wrote.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    written = 0
    fingering_added = 0
    system_text_added = 0
    refused: Counter = Counter()

    for index, source in enumerate(sources):
        stem = f"{source.parent.name}_aug"
        augmented_dir = out_root / stem
        augmented_dir.mkdir(parents=True, exist_ok=True)
        augmented_xml = augmented_dir / f"{stem}.musicxml"
        counts = augment(source, augmented_xml, fingering, system_text, seed + index)
        try:
            annotate(augmented_xml, augmented_dir)
        except Unrenderable as reason:
            refused[reason.reason] += 1
            continue
        written += 1
        fingering_added += counts["fingering"]
        system_text_added += counts["system_text"]

    return {
        "written": written,
        "fingering_added": fingering_added,
        "system_text_added": system_text_added,
        "refused": dict(refused),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("one", help="Augment and write a single MusicXML file (no render).")
    one.add_argument("--source", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--fingering", type=int, default=6)
    one.add_argument("--system-text", type=int, default=2)
    one.add_argument("--seed", type=int, default=0)

    batch = sub.add_parser("batch", help="Augment and render a batch of scores.")
    batch.add_argument("--sources", type=Path, required=True, help="Dir of .render.musicxml under score subfolders.")
    batch.add_argument("--count", type=int, default=80)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--fingering", type=int, default=4)
    batch.add_argument("--system-text", type=int, default=2)
    batch.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "one":
        counts = augment(args.source, args.out, args.fingering, args.system_text, args.seed)
        print(f"wrote {args.out}: {counts['fingering']} fingering, {counts['system_text']} system-text directions added")
    else:
        sources = sorted(args.sources.rglob("*.render.musicxml"))
        rng = random.Random(args.seed)
        rng.shuffle(sources)
        sources = sources[: args.count]
        if not sources:
            raise SystemExit(f"No .render.musicxml under {args.sources}")
        result = build_batch(sources, args.out, args.fingering, args.system_text, args.seed)
        print(
            f"{result['written']}/{len(sources)} scores rendered, "
            f"{result['fingering_added']} fingering + {result['system_text_added']} "
            "system-text boxes added"
        )
        if result["refused"]:
            print(f"  refused: {result['refused']}")


if __name__ == "__main__":
    main()
