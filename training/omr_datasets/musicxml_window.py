"""
Cut a renderable window out of a source MusicXML part.

PDMX builds its training images by regenerating MusicXML *from the tokens* and rendering
that. Tokens carry no beams or stems, so the renderer supplies its own - and any beam or
stem label taken from the source would then disagree with the picture exactly where the
engraving departs from the rule, which is the whole signal the notation heads exist to
learn. Rendering the source window instead removes that: the image shows what the score
actually says.

Two things make a mid-score window renderable on its own. It needs the clef, key,
divisions and time signature in force where it starts - a window beginning at measure 40
inherits them from earlier and would otherwise render with no clef at all. And it must be
a complete `score-partwise` document rather than a fragment.

The prevailing attributes are accumulated element by element rather than by copying the
last `<attributes>` block whole, because those blocks are partial: a mid-score block that
changes only the key does not restate the clef, so taking the most recent one would lose
it.
"""

import copy
import xml.etree.ElementTree as ET

#: Attribute children that carry forward until something changes them, in the order
#: MusicXML requires them to appear.
_CARRIED = ("divisions", "key", "time", "staves", "clef")


def prevailing_attributes(part: ET.Element, upto: int) -> ET.Element | None:
    """The clef, key, divisions and time in force just before measure index `upto`.

    Later declarations replace earlier ones of the same kind, except clefs, which are kept
    per staff number so a grand staff does not lose one of its two.
    """
    latest: dict[str, ET.Element] = {}
    clefs: dict[str, ET.Element] = {}

    for measure in part.findall("measure")[:upto]:
        for attributes in measure.findall("attributes"):
            for child in attributes:
                if child.tag == "clef":
                    clefs[child.get("number") or "1"] = child
                elif child.tag in _CARRIED:
                    latest[child.tag] = child

    if not latest and not clefs:
        return None

    merged = ET.Element("attributes")
    for tag in _CARRIED:
        if tag == "clef":
            for number in sorted(clefs):
                merged.append(copy.deepcopy(clefs[number]))
        elif tag in latest:
            merged.append(copy.deepcopy(latest[tag]))
    return merged


def extract_window(part: ET.Element, start: int, end: int) -> ET.Element | None:
    """A one-part score holding measures `start:end`, able to render on its own.

    Returns None for an empty range rather than an empty score, so a caller cannot render
    a blank image and pair it with a non-empty token sequence.
    """
    measures = part.findall("measure")[start:end]
    if not measures:
        return None

    score = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = ""

    single = ET.SubElement(score, "part", id="P1")
    for offset, measure in enumerate(measures):
        copied = copy.deepcopy(measure)
        if offset == 0:
            _ensure_context(copied, prevailing_attributes(part, start))
        single.append(copied)
    return score


def _ensure_context(measure: ET.Element, carried: ET.Element | None) -> None:
    """Put the inherited attributes at the front of the window's first measure.

    Anything the measure already states wins: a window that starts exactly where the clef
    changes must keep the new clef, not the one it inherited.
    """
    if carried is None:
        return
    own = measure.find("attributes")
    if own is None:
        measure.insert(0, carried)
        return

    stated = {child.tag for child in own if child.tag != "clef"}
    stated_clefs = {child.get("number") or "1" for child in own.findall("clef")}
    for child in reversed(list(carried)):
        if child.tag == "clef":
            if (child.get("number") or "1") in stated_clefs:
                continue
        elif child.tag in stated:
            continue
        own.insert(0, child)


def measure_count(part: ET.Element) -> int:
    return len(part.findall("measure"))
