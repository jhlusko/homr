"""
Syllable boxes from MuseScore, in the pixels of the image MuseScore drew.

27.42 settled that lyrics need their own OCR pass rather than a head, and that the stage
attaching syllables to notes needs positional supervision the scanned corpus cannot give -
the published MusicXML carries no `default-x` on notes or lyrics. Boxes have to come from a
renderer, and 27.25 says which renderer: the one that drew the image. Boxes from one
engraving over an image from another is the mismatch it exists to prevent.

27.43 established MuseScore emits what is needed - one `<path class="Lyrics">` per syllable,
not per glyph - and this module turns that into training data. The SVG and the PNG are
exports of the same layout, verified by drawing the boxes on the raster and looking at them:
fifteen tight boxes around fifteen syllables, with the hyphens correctly outside.

**The join is ordinal and the count check is load-bearing.** MuseScore's SVG carries no
element identity - nothing links a Lyrics path back to its `<lyric>` - so the only available
pairing is reading order. That is sound, because syllables do not overlap and read left to
right, but it means a miscount silently shifts every syllable onto the wrong box. It is the
same shape as 27.11, the sidecar substitution and the shared-tree bug in `lieder_voice`, so
it refuses rather than guesses.

Rendering needs `xvfb-run`; the AppImage aborts under `QT_QPA_PLATFORM=offscreen`.
"""

# flake8: noqa: T201

import argparse
import collections
import json
import multiprocessing
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2

#: MuseScore names elements by type in the SVG's class attribute.
LYRIC_CLASS = "Lyrics"

#: The melisma extender lines. Not syllables, but the resolve stage wants them - they mark
#: where one syllable is held across several notes, which is 32.9% of them (27.42).
EXTENDER_CLASS = "LyricsLineSegment"

#: Every class MuseScore draws as text. A page carries far more text than lyrics - title,
#: composer, tempo, dynamics, staff and system directions, fingerings, rehearsal marks - and
#: an OCR pass cropping a band under a staff will meet them whether or not it expects to.
#: 27.40 caught "sempre legato" sitting exactly where the lyric crop lands.
#:
#: homr represents none of them. Its six fields are rhythm, pitch, lift, articulation, slur
#: and position; the dynamics vocabulary exists but is commented out
#: (`homr/transformer/vocabulary.py`), and tempo, rehearsal marks, fingerings and titles were
#: never there. So typed text is capability homr does not currently have, not a duplicate of
#: something it does.
TEXT_CLASSES = frozenset(
    {
        "Lyrics",
        "Dynamic",
        "Tempo",
        "StaffText",
        "SystemText",
        "Expression",
        "Text",
        "InstrumentName",
        "MeasureNumber",
        "RehearsalMark",
        "Fingering",
        "Harmony",
    }
)

#: Classes whose path count can be checked against the source score, so the reading-order
#: join is verifiable for them. Measured on two full scores: Lyrics 134/134 and 155/155,
#: Dynamic 28/28.
#:
#: The direction texts are deliberately absent. MuseScore decides whether a `<words>`
#: becomes a Tempo, a StaffText or an Expression, and the MusicXML does not record which -
#: 21 rendered against 23 source `<words>` on one score. Their boxes are still extracted;
#: only the pairing to a string is withheld, because an unverifiable join is the thing this
#: module exists to avoid.
VERIFIABLE_CLASSES = frozenset({"Lyrics", "Dynamic"})

#: Rendering resolution. 150 gives roughly a 1240x1754 A4 page, close to the scanned
#: corpus, so the curriculum's two stages are not also a resolution change.
DPI = 150


class Unrenderable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


@dataclass(frozen=True)
class Syllable:
    text: str
    box: Box
    #: 'single', 'begin', 'middle' or 'end' - where the syllable sits inside its word.
    syllabic: str
    #: Which verse, so several lines of text under one staff stay separable.
    verse: str


def render(score: Path, out_dir: Path, dpi: int = DPI) -> tuple[list[Path], list[Path]]:
    """Export a score to SVG and PNG, one file per page, and return both lists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / score.stem
    for suffix in (".svg", ".png"):
        result = subprocess.run(
            ["xvfb-run", "-a", "mscore", "-r", str(dpi), "--export-to", str(stem) + suffix,
             str(score)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise Unrenderable(f"mscore failed on {suffix}: {result.stderr.strip()[:120]}")

    svgs = sorted(out_dir.glob(f"{score.stem}-*.svg"))
    pngs = sorted(out_dir.glob(f"{score.stem}-*.png"))
    if not svgs or len(svgs) != len(pngs):
        raise Unrenderable(f"{len(svgs)} svg pages against {len(pngs)} png pages")
    return svgs, pngs


def _path_bounds(data: str) -> tuple[float, float, float, float]:
    """The extent of one SVG path's coordinates.

    Control points of a bezier can sit slightly outside the drawn curve, so this is a
    marginal over-estimate rather than the exact glyph extent. For a text box that is the
    safe direction - a crop that clips a letter is worse than one with a pixel of margin.
    """
    xs = [float(value) for value in re.findall(r"[-\d.]+(?=,)", data)]
    ys = [float(value) for value in re.findall(r",([-\d.]+)", data)]
    if not xs or not ys:
        raise Unrenderable("path with no coordinates")
    return min(xs), min(ys), max(xs), max(ys)


def _scale(svg: str, image_width: int, image_height: int) -> tuple[float, float]:
    box = re.search(r'viewBox="([^"]+)"', svg)
    if not box:
        raise Unrenderable("svg has no viewBox")
    _, _, width, height = (float(value) for value in box.group(1).split())
    if width <= 0 or height <= 0:
        raise Unrenderable("svg viewBox has no extent")
    return image_width / width, image_height / height


def _lines(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes into lines of text, top line first, each read left to right.

    Bucketing by `top // height` was the first attempt and it is wrong: syllables on one
    line differ in top by several pixels - a capital or an accent raises it - so a line
    straddles a bucket boundary and splits in two. Grouping by whether boxes actually
    overlap vertically does not care where the boundaries fall.
    """
    if not boxes:
        return []
    heights = sorted(box.bottom - box.top for box in boxes)
    tolerance = max(1, heights[len(heights) // 2]) // 2

    lines: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: b.bottom):
        if lines and abs(box.bottom - lines[-1][-1].bottom) <= tolerance:
            lines[-1].append(box)
        else:
            lines.append([box])
    return [sorted(line, key=lambda b: b.left) for line in lines]


def boxes_of_class(svg: str, name: str, scale: tuple[float, float]) -> list[Box]:
    """Every box of one element class, in image pixels, in reading order.

    Reading order is line by line down the page, then left to right - which for lyrics
    means verse 1 entire, then verse 2. `source_syllables` orders itself to match.
    """
    sx, sy = scale
    found = []
    for data in re.findall(rf'<path class="{name}" d="([^"]+)"', svg):
        left, top, right, bottom = _path_bounds(data)
        found.append(
            Box(int(left * sx), int(top * sy), int(right * sx) + 1, int(bottom * sy) + 1)
        )
    return [box for line in _lines(found) for box in line]


def typed_boxes(svg: str, scale: tuple[float, float]) -> dict[str, list[Box]]:
    """Every text box on the page, grouped by what MuseScore says it is.

    The typing is free - it is already in the SVG's class attribute - which is why a pass
    that reads all page text costs no more annotation than one that reads only lyrics.
    """
    found = {}
    for name in sorted(TEXT_CLASSES):
        boxes = boxes_of_class(svg, name, scale)
        if boxes:
            found[name] = boxes
    return found


def source_dynamics(score: Path) -> list[str]:
    """Every dynamic marking, in document order.

    Rendered one path per element, and the count matches the source exactly, so these join
    the same checkable way lyrics do.
    """
    root = ET.parse(score).getroot()
    marks = []
    for element in root.iter("dynamics"):
        names = [child.tag for child in element]
        if names:
            marks.append("".join(names))
    return marks


def source_syllables(score: Path) -> list[tuple[str, str, str]]:
    """Every syllable as (text, syllabic, verse), ordered the way the page reads.

    **Not document order.** MusicXML interleaves verses note by note - note 1 verse 1, note
    1 verse 2, note 2 verse 1 - while the engraving puts each verse on its own line, so the
    page reads verse 1 entire and then verse 2. Pairing document order against visual order
    would misalign every syllable on a score with more than one verse, and 12.6% of
    lyric-bearing notes carry two to four (27.42).
    """
    root = ET.parse(score).getroot()
    found = []
    for part in root.findall("part"):
        for position, note in enumerate(part.iter("note")):
            for lyric in note.findall("lyric"):
                text = (lyric.findtext("text") or "").strip()
                if text:
                    verse = lyric.get("number", "1")
                    found.append(
                        (verse, position, text, lyric.findtext("syllabic") or "single")
                    )
    found.sort(key=lambda entry: (entry[0], entry[1]))
    return [(text, syllabic, verse) for verse, _, text, syllabic in found]


def pair(score: Path, svg_path: Path, png_path: Path) -> list[Syllable]:
    """Attach each rendered box to the syllable it draws.

    Refuses on a count mismatch. Reading order is the only join MuseScore's SVG allows, and
    under a mismatch it puts every syllable after the discrepancy on the wrong box - which
    is exactly the failure that is invisible downstream.
    """
    image = cv2.imread(str(png_path))
    if image is None:
        raise Unrenderable(f"cannot read {png_path}")

    svg = svg_path.read_text(encoding="utf-8")
    scale = _scale(svg, image.shape[1], image.shape[0])
    boxes = boxes_of_class(svg, LYRIC_CLASS, scale)
    syllables = source_syllables(score)

    if len(boxes) != len(syllables):
        raise Unrenderable(f"{len(boxes)} rendered boxes against {len(syllables)} syllables")

    return [
        Syllable(text, box, syllabic, verse)
        for box, (text, syllabic, verse) in zip(boxes, syllables)
    ]


def annotate(score: Path, out_dir: Path, dpi: int = DPI) -> dict:
    """Render one system and write everything readable about its text.

    The lyric boxes are paired to their syllables; every other text class is recorded as
    boxes alone, because only Lyrics and Dynamic have a count that can be checked against
    the source (27.44). Recording the unpaired classes anyway is deliberate: detection and
    classification need a box and a type, not a string.
    """
    svgs, pngs = render(score, out_dir, dpi)
    if len(svgs) != 1:
        # `pair` matches one page's boxes against the whole score's syllables, so a score
        # spilling onto a second page would pair the first page's boxes against all of
        # them. These inputs are single systems and should never spill.
        raise Unrenderable(f"expected one page, rendered {len(svgs)}")

    image = cv2.imread(str(pngs[0]))
    if image is None:
        raise Unrenderable(f"cannot read {pngs[0]}")
    svg = svgs[0].read_text(encoding="utf-8")
    scale = _scale(svg, image.shape[1], image.shape[0])

    syllables = pair(score, svgs[0], pngs[0])
    record = {
        "image": pngs[0].name,
        "width": image.shape[1],
        "height": image.shape[0],
        "lyrics": [
            {"text": s.text, "syllabic": s.syllabic, "verse": s.verse, **s.box.to_dict()}
            for s in syllables
        ],
        "text_boxes": {
            name: [box.to_dict() for box in boxes]
            for name, boxes in typed_boxes(svg, scale).items()
            if name != LYRIC_CLASS
        },
        "extenders": [box.to_dict() for box in boxes_of_class(svg, EXTENDER_CLASS, scale)],
    }
    (out_dir / f"{score.stem}.boxes.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return record


def _annotate_one(job: tuple[Path, Path, int]) -> tuple[str, int, int]:
    """Returns (reason or '', syllables, other text boxes). Runs in a worker process."""
    score, out_root, dpi = job
    try:
        record = annotate(score, out_root / score.stem, dpi)
    except (Unrenderable, subprocess.TimeoutExpired, OSError) as reason:
        return getattr(reason, "reason", str(reason))[:48], 0, 0
    other = sum(len(boxes) for boxes in record["text_boxes"].values())
    return "", len(record["lyrics"]), other


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scores", type=Path, required=True, help="A dir of .musicxml systems.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--workers", type=int, default=1, help="Renders in parallel; each is its own mscore."
    )
    args = parser.parse_args()

    scores = sorted(args.scores.rglob("*.musicxml"))
    if not scores:
        raise SystemExit(f"No .musicxml under {args.scores}")
    jobs = [(score, args.out, args.dpi) for score in scores]

    done = syllables = other = 0
    reasons: collections.Counter[str] = collections.Counter()
    with multiprocessing.Pool(max(1, args.workers)) as pool:
        for reason, found, boxes in pool.imap_unordered(_annotate_one, jobs, chunksize=4):
            if reason:
                reasons[reason] += 1
                continue
            done += 1
            syllables += found
            other += boxes

    print(f"{done:,} of {len(scores):,} systems annotated")
    print(f"  {syllables:,} syllables paired to boxes")
    print(f"  {other:,} other text boxes recorded without a string (27.44)")
    for reason, count in reasons.most_common(6):
        print(f"  {count:,} refused: {reason}")


if __name__ == "__main__":
    main()
