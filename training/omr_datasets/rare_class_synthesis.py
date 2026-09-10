"""
Manufacture Fingering training examples, by rendering real notation that just happens
not to occur often - not fabricated pixels.

27.89 fixed the detector's within-page positive sampling and found it could not move
Fingering off 0% precision: it occurs so rarely in the source corpus (12 boxes across
2,847 pages, 27.68) that no amount of resampling existing examples manufactures a signal
that was never there. `musescore_boxes.py`'s own render pipeline is the way out - it draws
whatever the MusicXML says to draw, so a `<fingering>` element added to a note is rendered
by the exact same MuseScore binary and turned into a box by the exact same SVG-class
extraction as every other page in the corpus (27.25's rule: boxes from the renderer that
drew the image). This is synthetic in the sense that the annotation was not in the
original score, not in the sense of a faked image - the pixels are real MuseScore
engraving of real MusicXML content.

27.90 built this alongside an equivalent SystemText injection (`<direction
system="only-top">`), on the hypothesis that `system="only-top"` was what MuseScore's own
SystemText-vs-StaffText choice keyed off. 27.92 found the result on a real retrain:
Fingering moved off 0% (18.8% F1), SystemText stayed at exactly 0% despite a comparable
injection, and every other class got measurably worse - a mixed result attributed to
27.89's class-balanced sampler drawing two classes' positive centres disproportionately
from the same 79 synthetic pages. Decided to fold SystemText into StaffText
(`detector_masks.CLASS_ALIASES`) rather than keep spending training data and attention on
a class this detector cannot resolve; the SystemText injection code is removed from this
module for the same reason - it worked as intended (real, correctly-classified SystemText
boxes) and still did not help, so keeping it around invites reaching for it again on the
same premise that already failed.
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


def _notes(root: ET.Element) -> list[ET.Element]:
    """Every real (non-rest) note across every part, in document order."""
    found = []
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                if note.find("rest") is None and note.find("pitch") is not None:
                    found.append(note)
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


def augment(source: Path, target: Path, fingering: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    tree = ET.parse(source)
    root = tree.getroot()
    counts = {"fingering": inject_fingering(root, fingering, rng)}
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return counts


def verify_svg_classes(svg_path: Path) -> Counter:
    """What MuseScore actually drew, by SVG class - a check to run before trusting a new
    injection at scale, the way this module's own SystemText attempt should have been
    trusted less readily (27.92)."""
    text = svg_path.read_text(encoding="utf-8")
    return Counter(cls for cls in ("Fingering", "StaffText", "Expression", "Tempo") if f'class="{cls}"' in text)


def build_batch(sources: list[Path], out_root: Path, fingering: int, seed: int) -> dict:
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
    refused: Counter = Counter()

    for index, source in enumerate(sources):
        stem = f"{source.parent.name}_aug"
        augmented_dir = out_root / stem
        augmented_dir.mkdir(parents=True, exist_ok=True)
        augmented_xml = augmented_dir / f"{stem}.musicxml"
        counts = augment(source, augmented_xml, fingering, seed + index)
        try:
            annotate(augmented_xml, augmented_dir)
        except Unrenderable as reason:
            refused[reason.reason] += 1
            continue
        written += 1
        fingering_added += counts["fingering"]

    return {"written": written, "fingering_added": fingering_added, "refused": dict(refused)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("one", help="Augment and write a single MusicXML file (no render).")
    one.add_argument("--source", type=Path, required=True)
    one.add_argument("--out", type=Path, required=True)
    one.add_argument("--fingering", type=int, default=6)
    one.add_argument("--seed", type=int, default=0)

    batch = sub.add_parser("batch", help="Augment and render a batch of scores.")
    batch.add_argument("--sources", type=Path, required=True, help="Dir of .render.musicxml under score subfolders.")
    batch.add_argument("--count", type=int, default=80)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument("--fingering", type=int, default=4)
    batch.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "one":
        counts = augment(args.source, args.out, args.fingering, args.seed)
        print(f"wrote {args.out}: {counts['fingering']} fingering directions added")
    else:
        sources = sorted(args.sources.rglob("*.render.musicxml"))
        rng = random.Random(args.seed)
        rng.shuffle(sources)
        sources = sources[: args.count]
        if not sources:
            raise SystemExit(f"No .render.musicxml under {args.sources}")
        result = build_batch(sources, args.out, args.fingering, args.seed)
        print(f"{result['written']}/{len(sources)} scores rendered, {result['fingering_added']} fingering boxes added")
        if result["refused"]:
            print(f"  refused: {result['refused']}")


if __name__ == "__main__":
    main()
