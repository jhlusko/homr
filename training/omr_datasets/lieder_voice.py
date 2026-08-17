"""
Put the voice part and its lyrics back into OLiMPiC's per-system scores.

27.40 fixed the images: extending the system boxes recovers the voice staff and the lyrics
printed under it. The labels stayed wrong. OLiMPiC calls `MxlFile.get_piano_part()` and
throws the rest of the score away before it ever slices, and `Pruner` strips `<lyric>` on
top of that - so its MusicXML describes a piano reduction of a picture that now shows a
singer.

27.39 assumed recovering the voice meant re-running OLiMPiC's build on the unreduced score,
which needs MuseScore and needs its page layout reproduced exactly, since the systems come
from `<print new-system>` markers that MuseScore writes at export. Two measurements made
that unnecessary:

    published lc5837811.mxl   P1 Chant/Voice  49 measures, numbered 0..48, 169 lyrics
                              P2 Piano        49 measures, numbered 0..48
    olimpic 5837811           16 systems covering measures 0..48, contiguous

**OLiMPiC's slicing preserves the original measure numbers.** p1-s1 holds measures 0-2,
p1-s2 holds 3-5, p2-s1 holds 12-14. So a system is identified by a measure range, and the
same range read out of the published score gives the voice part for exactly that system.
The join is arithmetic on measure numbers, not geometry, and OpenScore Lieder publishes
`.mxl` for all 1,462 of its scores - no MuseScore anywhere in the path.

The correspondence risk 27.11 warns about is the whole risk here, so the range is checked
rather than trusted: every measure the sample names must exist in the voice part, and the
count must match. A voice part silently a measure short would put every lyric after it
under the wrong note.
"""

# flake8: noqa: T201

import argparse
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

#: What OLiMPiC's `resolve_piano_part_id` matches on, kept identical so the part this
#: module discards is exactly the part OLiMPiC keeps. Diverging would pair a voice from one
#: reading of the score with a piano from another.
PIANO_INSTRUMENTS = frozenset(
    {"Piano", "Grand Piano", "Acoustic Grand Piano", "Harpsichord", "Pianoforte", "Piano (2)"}
)
PIANO_PART_NAMES = frozenset({"Pianoforte"})


class Unjoinable(Exception):
    """The voice cannot be attached to this system, with the reason why."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Joined:
    score: ET.ElementTree
    measures: tuple[str, ...]
    lyrics: int


def read_mxl(path: Path) -> ET.ElementTree:
    """Read a compressed MusicXML, taking the score rather than the container."""
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith(".xml") and not name.startswith("META-INF")
        ]
        if not names:
            raise Unjoinable("no score inside the mxl container")
        return ET.ElementTree(ET.fromstring(archive.read(names[0])))


def piano_part_id(score: ET.ElementTree) -> str | None:
    part_list = score.getroot().find("part-list")
    for entry in part_list.findall("score-part") if part_list is not None else []:
        instrument = entry.findtext("score-instrument/instrument-name")
        if instrument in PIANO_INSTRUMENTS or entry.findtext("part-name") in PIANO_PART_NAMES:
            return entry.get("id")
    return None


def voice_parts(score: ET.ElementTree) -> list[ET.Element]:
    """Every part that is not the piano.

    Plural because Lieder duets exist, and taking only the first would drop a singer
    without saying so.
    """
    piano = piano_part_id(score)
    return [part for part in score.getroot().findall("part") if part.get("id") != piano]


def measure_numbers(part: ET.Element) -> list[str]:
    return [measure.get("number", "") for measure in part.findall("measure")]


def system_measures(sample: Path) -> tuple[str, ...]:
    """The measure numbers one of OLiMPiC's per-system scores covers."""
    root = ET.parse(sample).getroot()
    numbers = tuple(measure.get("number", "") for measure in root.iter("measure"))
    if not numbers:
        raise Unjoinable("system score has no measures")
    return numbers


def slice_part(part: ET.Element, wanted: tuple[str, ...]) -> ET.Element:
    """The named measures of one part, as a part of its own.

    Refuses rather than returns a short part. A voice missing one measure would shift every
    lyric after it onto a different note, and nothing downstream could tell.
    """
    by_number = {measure.get("number", ""): measure for measure in part.findall("measure")}
    missing = [number for number in wanted if number not in by_number]
    if missing:
        raise Unjoinable(f"voice part is missing measures {missing[:3]}")

    sliced = ET.Element("part", {"id": part.get("id", "P1")})
    for number in wanted:
        sliced.append(by_number[number])
    return sliced


def _carry_attributes(part: ET.Element, source: ET.Element, wanted: tuple[str, ...]) -> None:
    """Give the slice the clef, key and divisions in force where it starts.

    A system that begins mid-score inherits an attributes header in the engraving, and
    OLiMPiC re-emits one for the same reason. Without it the slice is read in whatever
    default the parser assumes, which for a voice part means the wrong clef.
    """
    first = part.find("measure")
    if first is None or first.find("attributes") is not None:
        return

    running = ET.Element("attributes")
    for measure in source.findall("measure"):
        if measure.get("number", "") == wanted[0]:
            break
        for attributes in measure.findall("attributes"):
            for child in attributes:
                if child.tag in {"divisions", "key", "time", "clef", "staves"}:
                    running.append(child)
    if len(running):
        first.insert(0, running)


def join(sample: Path, full_score: ET.ElementTree) -> Joined:
    """Build a system score holding the voice parts and the piano, lyrics intact."""
    wanted = system_measures(sample)
    voices = voice_parts(full_score)
    if not voices:
        raise Unjoinable("no non-piano part in the published score")

    piano_system = ET.parse(sample).getroot()
    combined = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(combined, "part-list")

    published_list = full_score.getroot().find("part-list")
    published = {entry.get("id"): entry for entry in published_list.findall("score-part")}

    lyrics = 0
    for voice in voices:
        sliced = slice_part(voice, wanted)
        _carry_attributes(sliced, voice, wanted)
        lyrics += len(sliced.findall(".//lyric"))
        entry = published.get(voice.get("id"))
        if entry is not None:
            part_list.append(entry)
        combined.append(sliced)

    # The piano comes from OLiMPiC's own sample rather than being re-sliced, so this half
    # stays byte-for-byte what its published labels describe.
    for entry in piano_system.findall("part-list/score-part"):
        part_list.append(entry)
    for part in piano_system.findall("part"):
        combined.append(part)

    return Joined(ET.ElementTree(combined), wanted, lyrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--samples", type=Path, required=True, help="An olimpic samples dir.")
    parser.add_argument("--lieder", type=Path, required=True, help="Dir of lc<id>.mxl scores.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    joined = refused = with_lyrics = 0
    reasons: dict[str, int] = {}

    for score_dir in sorted(p for p in args.samples.iterdir() if p.is_dir()):
        mxl = args.lieder / f"lc{score_dir.name}.mxl"
        if not mxl.is_file():
            reasons["no published score"] = reasons.get("no published score", 0) + 1
            continue
        try:
            full = read_mxl(mxl)
        except (Unjoinable, zipfile.BadZipFile) as broken:
            reasons[str(broken)[:40]] = reasons.get(str(broken)[:40], 0) + 1
            continue

        for sample in sorted(score_dir.glob("*.musicxml")):
            try:
                result = join(sample, full)
            except Unjoinable as reason:
                reasons[reason.reason[:40]] = reasons.get(reason.reason[:40], 0) + 1
                refused += 1
                continue
            target = args.out / score_dir.name
            target.mkdir(parents=True, exist_ok=True)
            result.score.write(target / sample.name, encoding="utf-8", xml_declaration=True)
            joined += 1
            with_lyrics += 1 if result.lyrics else 0

    print(f"{joined:,} systems joined, {with_lyrics:,} of them carrying lyrics")
    if refused or reasons:
        for reason, count in sorted(reasons.items(), key=lambda pair: -pair[1])[:6]:
            print(f"  {count:,} refused: {reason}")


if __name__ == "__main__":
    main()
