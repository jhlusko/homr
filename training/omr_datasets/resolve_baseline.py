"""
How much of syllable-to-note attachment is just horizontal proximity?

27.45 named the resolve stage the risky component: attaching a recognised syllable to the
note it belongs under is the correspondence problem that has produced four defects in this
work already. Before building a model for it, the question Gate C asked about beams is worth
asking here - **what does the obvious rule already get right?** A learned resolver that
cannot beat "nearest note by x" is not worth its training time, and one that can needs to be
judged against that number rather than against zero.

The rule: a syllable attaches to the note whose horizontal centre is nearest its own. That is
what a reader does at a glance, and engraving is arranged to make it true - the syllable is
centred under its notehead.

Where it should fail is measurable in advance, and 27.42 measured it:

  * **melismas** - 32.9% of syllables are held across more than one note, so a syllable sits
    under the first of several and the rest carry none. Nearest-x will attach the *following*
    syllable to a note in the middle of a melisma.
  * **several verses** - 12.6% of lyric-bearing notes carry two to four, stacked vertically,
    which x alone cannot separate. Only verse 1 is scored here.

Ground truth comes from MuseScore's own render, so image and labels are one engraving (27.25).
Note boxes come from the SVG rather than the `.boxes.json`, which records only text classes.
**Only the voice staff's notes count**, and they are separated geometrically: a Lieder system
is voice, then lyrics, then piano, so the voice notes are the ones above the lyrics.

That separation assumes **one** vocal staff, and not every score has one. A Lied for two
voices puts verse-1 syllables under both staves, at two different heights, and a single
"above the lyrics" band then spans the whole system - measured on the first score tried, 777
to 2395 pixels, swallowing the piano. Those systems are excluded rather than handled, and
counted, because the rule this file exists to measure is about horizontal position and a
second staff is a vertical question. Scoring them under a broken filter would report the
filter's failure as the rule's.

The pairing is ordinal - the k-th lyric-bearing note takes the k-th syllable - and refuses on
a count mismatch, the same guard as everywhere else here. A shifted pairing would make the
rule look wrong where in fact the labels were.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


def _bounds(data: str) -> tuple[float, float, float, float]:
    xs = [float(value) for value in re.findall(r"[-\d.]+(?=,)", data)]
    ys = [float(value) for value in re.findall(r",([-\d.]+)", data)]
    return (min(xs), min(ys), max(xs), max(ys)) if xs and ys else (0.0, 0.0, 0.0, 0.0)


#: A MuseScore glyph is either drawn in place, like a lyric, or drawn at the origin and
#: moved by a transform, like a notehead. Both appear in the same file, and reading a
#: transformed path's `d` as though it were absolute puts every notehead at the top left.
_ELEMENT = re.compile(r'<path class="([A-Za-z]+)"([^>]*)>')
_TRANSLATE = re.compile(r"matrix\(([-\d.]+),[-\d.]+,[-\d.]+,([-\d.]+),([-\d.]+),([-\d.]+)\)")
_PATH_DATA = re.compile(r'd="([^"]+)"')


def placed_boxes(svg: str, name: str) -> list[tuple[float, float, float, float]]:
    """Every box of one class, in the SVG's own units, transform applied."""
    boxes = []
    for element, attributes in _ELEMENT.findall(svg):
        if element != name:
            continue
        data = _PATH_DATA.search(attributes)
        if not data:
            continue
        left, top, right, bottom = _bounds(data.group(1))
        moved = _TRANSLATE.search(attributes)
        if moved:
            scale_x, scale_y, offset_x, offset_y = (float(v) for v in moved.groups())
            left, right = offset_x + left * scale_x, offset_x + right * scale_x
            top, bottom = offset_y + top * scale_y, offset_y + bottom * scale_y
        boxes.append((left, top, right, bottom))
    return boxes


def note_centres(svg_path: Path, width: int, height: int, above: float) -> list[float]:
    """Horizontal centres of the noteheads above `above`, left to right.

    `above` is the top of the lyric band. A Lieder system is voice, then its lyrics, then
    the piano, so everything above the lyrics belongs to the singer - which is the staff the
    syllables attach to. Without this filter the piano's notes, several times as numerous,
    would be candidates and the rule would be measured against a question nobody asks.
    """
    svg = svg_path.read_text(encoding="utf-8")
    box = re.search(r'viewBox="([^"]+)"', svg)
    if not box:
        return []
    _, _, view_width, view_height = (float(value) for value in box.group(1).split())
    sx, sy = width / view_width, height / view_height

    centres = []
    for left, _, right, bottom in placed_boxes(svg, "Note"):
        if bottom * sy < above:
            centres.append((left + right) / 2 * sx)
    return sorted(centres)


def lyric_bearing_positions(score: Path) -> list[int]:
    """Positions, among the voice part's sounding notes, of those carrying a verse-1 syllable.

    Rests and chord members are skipped: a rest bears no syllable, and a chord's second
    notehead is not a separate note to attach to.
    """
    root = ET.parse(score).getroot()
    for part in root.findall("part"):
        if not part.findall(".//lyric"):
            continue
        positions, index = [], 0
        for note in part.iter("note"):
            if note.find("chord") is not None or note.find("rest") is not None:
                continue
            if any(
                (lyric.findtext("text") or "").strip()
                for lyric in note.findall("lyric")
                if lyric.get("number", "1") == "1"
            ):
                positions.append(index)
            index += 1
        return positions
    return []


def nearest(x: float, centres: list[float]) -> int:
    return min(range(len(centres)), key=lambda index: abs(centres[index] - x))


@dataclass
class Agreement:
    correct: int = 0
    total: int = 0
    offsets: collections.Counter[int] = field(default_factory=collections.Counter)
    #: Kept separately because 27.42 predicts melismas are where this rule breaks, and a
    #: single accuracy would not say whether the prediction held.
    melismatic: int = 0
    melismatic_correct: int = 0

    def observe(self, chosen: int, true_index: int, held: bool) -> None:
        self.total += 1
        self.correct += chosen == true_index
        self.offsets[chosen - true_index] += 1
        if held:
            self.melismatic += 1
            self.melismatic_correct += chosen == true_index

    def describe(self) -> str:
        if not self.total:
            return "nothing to score"
        near = sum(count for offset, count in self.offsets.items() if abs(offset) <= 1)
        plain = self.total - self.melismatic
        plain_correct = self.correct - self.melismatic_correct
        lines = [
            f"syllables scored: {self.total:,}",
            f"  nearest-x picks the right note: {self.correct:,} ({self.correct / self.total:.1%})",
            f"  within one note either side:    {near:,} ({near / self.total:.1%})",
            "",
            "  split by whether the syllable is held across notes (27.42 predicts this is",
            "  where the rule breaks):",
            f"    one note only: {plain_correct:,}/{plain:,}"
            f" ({plain_correct / max(1, plain):.1%})",
            f"    held (melisma): {self.melismatic_correct:,}/{self.melismatic:,}"
            f" ({self.melismatic_correct / max(1, self.melismatic):.1%})",
            "",
            "  offset from the true note (chosen minus true):",
        ]
        for offset, count in sorted(self.offsets.items())[:9]:
            lines.append(f"    {offset:+d}: {count:,} ({count / self.total:.1%})")
        return "\n".join(lines)


def single_lyric_line(syllables: list[dict]) -> bool:
    """Whether every syllable sits on one line of text.

    A second vocal staff puts verse-1 syllables at a second height, and the vertical
    separation this file relies on stops meaning anything. One syllable's height is the
    tolerance: within a line, tops differ by a few pixels for capitals and accents.
    """
    if len(syllables) < 2:
        return True
    heights = sorted(box["bottom"] - box["top"] for box in syllables)
    tolerance = max(4, heights[len(heights) // 2])
    tops = [box["top"] for box in syllables]
    return max(tops) - min(tops) <= tolerance


def lyric_lines(syllables: list[dict]) -> list[list[dict]]:
    """Group syllables into lines of text, topmost first, each left to right.

    27.52 found 41% of systems carry more than one line - a second vocal staff, or a second
    verse engraved below the first - and excluded them, because "the voice notes are above
    the lyrics" stops meaning anything with two bands of lyrics. This is the grouping that
    lets them be scored instead of skipped.

    Grouped by vertical overlap rather than by bucketing the top coordinate, for the reason
    `musescore_boxes._lines` records: a capital or an accent raises a box's top by several
    pixels, and a bucket boundary then splits one line in two.
    """
    if not syllables:
        return []
    heights = sorted(box["bottom"] - box["top"] for box in syllables)
    tolerance = max(4, heights[len(heights) // 2]) // 2

    lines: list[list[dict]] = []
    for box in sorted(syllables, key=lambda b: b["bottom"]):
        if lines and abs(box["bottom"] - lines[-1][-1]["bottom"]) <= tolerance:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [sorted(line, key=lambda b: b["left"]) for line in lines]


def parts_with_lyrics(score: Path) -> list[list[int]]:
    """Per lyric-carrying part, the positions of its verse-1 lyric-bearing notes.

    Document order, which is top-to-bottom on the page: MusicXML lists parts in score
    order, and an engraver puts the first part on the top staff. That is the assumption the
    line-to-part pairing rests on, and it is checked by count rather than trusted.
    """
    root = ET.parse(score).getroot()
    found = []
    for part in root.findall("part"):
        if not part.findall(".//lyric"):
            continue
        positions, index = [], 0
        for note in part.iter("note"):
            if note.find("chord") is not None or note.find("rest") is not None:
                continue
            if any(
                (lyric.findtext("text") or "").strip()
                for lyric in note.findall("lyric")
                if lyric.get("number", "1") == "1"
            ):
                positions.append(index)
            index += 1
        found.append(positions)
    return found


def score_system(
    record_path: Path, score: Path, agreement: Agreement, skips: collections.Counter | None = None
) -> bool:
    """Score one system; returns False when it had to be skipped.

    Skips are counted by reason. Nearly half the systems are excluded, and a headline
    accuracy over the remainder means nothing without knowing what left - a filter that
    quietly removed the hard cases would report the easy ones as the whole picture.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    syllables = [box for box in record["lyrics"] if box.get("verse", "1") == "1"]
    positions = lyric_bearing_positions(score)
    counts = skips if skips is not None else collections.Counter()
    if not syllables:
        counts["no verse-1 syllables"] += 1
        return False
    if len(syllables) != len(positions):
        counts["syllable count disagrees with the score"] += 1
        return False
    if not single_lyric_line(syllables):
        counts["more than one lyric line (second vocal staff)"] += 1
        return False

    top_of_lyrics = min(box["top"] for box in syllables)
    svg = record_path.parent / (record["image"][: -len(".png")] + ".svg")
    if not svg.is_file():
        counts["no svg"] += 1
        return False
    centres = note_centres(svg, record["width"], record["height"], top_of_lyrics)
    if not centres:
        counts["no notes above the lyrics"] += 1
        return False
    if positions[-1] >= len(centres):
        counts["fewer notes found than the score names"] += 1
        return False

    for index, (box, position) in enumerate(zip(syllables, positions)):
        # A syllable is held when the next one is more than one note further along.
        following = positions[index + 1] if index + 1 < len(positions) else position + 1
        held = following - position > 1
        chosen = nearest((box["left"] + box["right"]) / 2, centres)
        agreement.observe(chosen, position, held)
    return True


def notes_between(svg_path: Path, width: int, height: int, ceiling: float, floor_: float):
    """Horizontal centres of noteheads in a horizontal band, left to right.

    A staff's notes sit between the lyrics of the line above and the lyrics of its own line.
    For the topmost staff the ceiling is the page edge.
    """
    svg = svg_path.read_text(encoding="utf-8")
    box = re.search(r'viewBox="([^"]+)"', svg)
    if not box:
        return []
    _, _, view_width, view_height = (float(value) for value in box.group(1).split())
    sx, sy = width / view_width, height / view_height
    return sorted(
        (left + right) / 2 * sx
        for left, _, right, bottom in placed_boxes(svg, "Note")
        if ceiling < bottom * sy < floor_
    )


@dataclass
class LineCounts:
    """Whether vertical clustering finds as many lines as the score has lyric parts."""

    agreed: int = 0
    total: int = 0
    deltas: collections.Counter[int] = field(default_factory=collections.Counter)

    def observe(self, found: int, expected: int) -> None:
        self.total += 1
        self.agreed += found == expected
        self.deltas[found - expected] += 1

    def describe(self) -> str:
        if not self.total:
            return "no multi-line systems seen"
        parts = ", ".join(f"{d:+d}:{c:,}" for d, c in sorted(self.deltas.items())[:7])
        return (
            f"line count agrees with the score: {self.agreed:,}/{self.total:,} "
            f"({self.agreed / self.total:.1%})\n    found minus expected: {parts}"
        )


def score_multiline(
    record_path: Path, score: Path, agreement: Agreement, counts: LineCounts
) -> bool:
    """Score a system by assigning syllables to lines first, then notes within each line."""
    record = json.loads(record_path.read_text(encoding="utf-8"))
    syllables = [box for box in record["lyrics"] if box.get("verse", "1") == "1"]
    parts = parts_with_lyrics(score)
    if not syllables or not parts:
        return False

    lines = lyric_lines(syllables)
    counts.observe(len(lines), len(parts))
    if len(lines) != len(parts):
        return False
    if any(len(line) != len(positions) for line, positions in zip(lines, parts)):
        return False

    svg = record_path.parent / (record["image"][: -len(".png")] + ".svg")
    if not svg.is_file():
        return False

    ceiling = 0.0
    for line, positions in zip(lines, parts):
        top = min(box["top"] for box in line)
        centres = notes_between(svg, record["width"], record["height"], ceiling, top)
        ceiling = max(box["bottom"] for box in line)
        if not centres or positions[-1] >= len(centres):
            continue
        for index, (box, position) in enumerate(zip(line, positions)):
            following = positions[index + 1] if index + 1 < len(positions) else position + 1
            agreement.observe(
                nearest((box["left"] + box["right"]) / 2, centres), position,
                following - position > 1,
            )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--boxes", type=Path, required=True, help="A musescore_boxes out dir.")
    parser.add_argument("--scores", type=Path, required=True, help="Dir of joined .musicxml.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--multiline",
        action="store_true",
        help="Assign syllables to lines first, then notes within each line - so the 41%% of "
        "systems 27.52 skipped are scored instead (27.55).",
    )
    args = parser.parse_args()

    agreement = Agreement()
    counts = LineCounts()
    skips: collections.Counter[str] = collections.Counter()
    skipped = scored = 0
    records = sorted(args.boxes.glob("*/*.boxes.json"))
    if args.limit:
        records = records[: args.limit]

    for path in records:
        name = path.name[: -len(".boxes.json")]
        score = args.scores / f"{name}.musicxml"
        if not score.is_file():
            skips["no joined score on disk"] += 1
            skipped += 1
        elif args.multiline and score_multiline(path, score, agreement, counts):
            scored += 1
        elif args.multiline:
            skipped += 1
        elif score_system(path, score, agreement, skips):
            scored += 1
        else:
            skipped += 1

    print(agreement.describe())
    if args.multiline:
        print("\n  " + counts.describe())
    print(f"\n{scored:,} systems scored, {skipped:,} skipped:")
    for reason, count in skips.most_common():
        print(f"    {count:,}  {reason}")


if __name__ == "__main__":
    main()
