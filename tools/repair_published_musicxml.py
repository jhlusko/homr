"""Apply the two generator fixes to MusicXML that was already published.

Both defects were in `homr/music_xml_generator.py` and its inputs, so every artifact this
repository has ever rendered carries them - the `/ots-homr` galleries, the review sets, the
writeups' comparison scores. Re-running inference would fix them too, but it would also
change *which* examples they are, and these were curated: captions, manifests and prose
refer to what is in each file.

So this rewrites the published files in place instead, applying exactly what the fixed code
now emits:

**Beams above a note's flag count are removed.** Training masks every beam level a note's
duration cannot carry, so the head was never supervised there and its output is an unlearned
projection. `build_beams` wrote it anyway. An eighth note keeps `<beam number="1">` and
nothing more; a half note keeps none.

**Slur numbers are reallocated from an open-span stack.** A number must be unique across
every span open at that moment; the generator used the sidecar's slot, which is unique only
within one note, so two staves each opening their slot 1 both emitted `number="1"`. The
original number *is* the slot, so the span's identity is recoverable from the file: key on
(staff, original number) and reallocate.

Both rewrites are deterministic and checkable - `--check` re-audits without writing, and the
counts before and after are the test. Neither invents anything: no beam is added, no slur
endpoint moves, and nothing is repaired that was not provably wrong.

This is a repair of artifacts, not a re-render. A gallery regenerated against final models
is still required before release.
"""

# flake8: noqa: T201

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

#: MusicXML `<type>` -> how many flags that written value carries. Anything absent from
#: this table (whole, half, quarter, breve...) carries none and so can hold no beam.
FLAGS = {
    "eighth": 1,
    "16th": 2,
    "32nd": 3,
    "64th": 4,
    "128th": 5,
    "256th": 6,
    "512th": 7,
    "1024th": 8,
}


def _strip_untrained_beams(root: ET.Element) -> int:
    """Remove every `<beam>` at a level the note's own duration cannot carry."""
    removed = 0
    for note in root.iter("note"):
        allowed = FLAGS.get((note.findtext("type") or "").strip(), 0)
        for beam in list(note.findall("beam")):
            try:
                level = int(beam.get("number", "1"))
            except ValueError:
                level = 1
            if level > allowed:
                note.remove(beam)
                removed += 1
    return removed


def _renumber_slurs(root: ET.Element) -> int:
    """Reallocate slur numbers so no two spans open at once share one.

    Walks the document in order, which is the order the generator wrote and the order a
    reader consumes. `(staff, original number)` is the span's identity: the original number
    was the sidecar slot, which is exactly what the fixed generator now carries as the key.
    """
    changed = 0
    open_numbers: list[int] = []
    by_key: dict[tuple[str, str], list[int]] = {}
    for note in root.iter("note"):
        staff = note.findtext("staff") or "1"
        for slur in note.findall("./notations/slur"):
            original = slur.get("number", "1")
            key = (staff, original)
            if slur.get("type") == "stop":
                if by_key.get(key):
                    number = by_key[key].pop()
                    if number in open_numbers:
                        open_numbers.remove(number)
                elif open_numbers:
                    number = open_numbers.pop()
                else:
                    # A stop with nothing open - the generator emits a number anyway so
                    # the defect stays visible rather than vanishing. Match that.
                    number = 1
            else:
                number = 1
                while number in open_numbers:
                    number += 1
                open_numbers.append(number)
                by_key.setdefault(key, []).append(number)
            if str(number) != original:
                slur.set("number", str(number))
                changed += 1
    return changed


def audit(root: ET.Element) -> tuple[int, int]:
    """(slur-number collisions, beams above a note's flag count) - what is still wrong."""
    live: dict[str, int] = {}
    collisions = beams = 0
    for note in root.iter("note"):
        allowed = FLAGS.get((note.findtext("type") or "").strip(), 0)
        for beam in note.findall("beam"):
            try:
                level = int(beam.get("number", "1"))
            except ValueError:
                level = 1
            if level > allowed:
                beams += 1
        for slur in note.findall("./notations/slur"):
            number = slur.get("number", "1")
            if slur.get("type") == "start":
                live[number] = live.get(number, 0) + 1
                if live[number] > 1:
                    collisions += 1
            elif live.get(number):
                live[number] -= 1
    return collisions, beams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory to walk for .musicxml files")
    parser.add_argument("--check", action="store_true", help="audit only, write nothing")
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    touched = []
    for path in sorted(args.root.rglob("*.musicxml")):
        try:
            tree = ET.parse(path)
        except ET.ParseError as ex:
            print(f"  unreadable, skipped: {path} ({ex})")
            totals["unreadable"] += 1
            continue
        root = tree.getroot()
        before = audit(root)
        totals["files"] += 1
        totals["collisions_before"] += before[0]
        totals["beams_before"] += before[1]
        if args.check:
            continue

        removed = _strip_untrained_beams(root)
        renumbered = _renumber_slurs(root)
        after = audit(root)
        totals["beams_removed"] += removed
        totals["slurs_renumbered"] += renumbered
        totals["collisions_after"] += after[0]
        totals["beams_after"] += after[1]
        if removed or renumbered:
            tree.write(path, encoding="unicode", xml_declaration=True)
            touched.append((path, removed, renumbered))

    print(f"{totals['files']:,} files")
    print(
        f"  slur-number collisions   {totals['collisions_before']:,}"
        + ("" if args.check else f" -> {totals['collisions_after']:,}")
    )
    print(
        f"  beams above flag count   {totals['beams_before']:,}"
        + ("" if args.check else f" -> {totals['beams_after']:,}")
    )
    if not args.check:
        print(
            f"  rewrote {len(touched):,} files "
            f"({totals['beams_removed']:,} beams removed, "
            f"{totals['slurs_renumbered']:,} slur numbers changed)"
        )


if __name__ == "__main__":
    main()
