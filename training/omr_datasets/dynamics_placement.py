"""
Recover dynamics the same way 27.20 recovered slur placement: from the original
whole-score MusicXML, joined to the systemwise segments by position.

27.94-27.97 built the whole dynamics strategy - detector, classifier, attachment rule,
structured head - on the assumption that `structured_notation_parser.py`'s
`NotationExtractor` would see `<direction>` elements when convert_ossq.py tokenises a
segment. Phase14 (28.1) found that assumption false: the "unaligned" segment MusicXML
convert_ossq.py actually reads carries zero `<direction>` elements anywhere in the corpus,
because the MuseScore round-trip that produces those segments drops them entirely - the
same loss `slur_placement.py`'s docstring already documented for slur placement, just
total here instead of partial.

The fix is the same shape as 27.20's: the dynamics are still on the original whole-score
`<score-id>.musicxml` (not `_cleaned`, which strips them too - see 28.1), and the only way
back to a segment is the positional join `slur_placement.py` already validated (`is_visible`,
`part_signature`) - a part's segments concatenate, in reading order, to the same visible
notes as the whole part reads note-for-note.

Deliberately reuses `slur_placement.py`'s alignment primitives rather than re-deriving
them. The join is the fragile, previously-broken part (27.20's docstring: broken five
times); nothing here should risk breaking it a sixth for the sake of not importing.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.structured_notation import DynamicMark
from training.omr_datasets.slur_placement import (
    concatenated,
    is_visible,
    part_signature,
    segments_of,
)
from training.omr_datasets.structured_notation_parser import NotationExtractor


def part_dynamics(part: ET.Element) -> list[DynamicMark]:
    """One dynamic mark per *visible* note in the part, document order.

    Filtered to visible notes rather than every note extract() sees, to match
    `part_signature`'s alignment unit - the positional join is only valid note-for-note
    against that same filter. Invisible notes still update the pending-dynamic state (an
    invisible note can still be the "next note" a direction attaches to in the source),
    they are just not written to the returned list.
    """
    extractor = NotationExtractor()
    marks: list[DynamicMark] = []
    for measure in part.findall("measure"):
        for child in measure:
            if child.tag == "direction":
                extractor.handle_direction(child)
            elif child.tag == "note":
                notation = extractor.extract(child)
                if is_visible(child):
                    marks.append(notation.dynamic)
    return marks


class DynamicsPlacementIndex:
    """Dynamics for every segment of one score, or nothing if it cannot be trusted.

    Structurally identical to `slur_placement.PlacementIndex` - built per score, sliced
    by segment rather than consumed as a running cursor, for the same reason: a system
    skipped for a crop mismatch still occupies notes in the whole score, and a cursor
    would silently misalign after the first skip.
    """

    def __init__(self, work: Path, score_id: str, whole: Path) -> None:
        self.slices: dict[tuple[int, int, int], list[DynamicMark]] = {}
        self.aligned_parts = 0
        self.skipped_parts = 0
        self._build(work, score_id, whole)

    def _build(self, work: Path, score_id: str, whole: Path) -> None:
        try:
            whole_parts = ET.parse(whole).getroot().findall("part")  # noqa: S314
        except (ET.ParseError, OSError):
            return
        segments = segments_of(work, score_id)
        if not segments:
            return

        for part_index, whole_part in enumerate(whole_parts):
            expected = part_signature(whole_part)
            found, _ = concatenated(segments, part_index)
            if len(expected) != len(found) or expected != found:
                self.skipped_parts += 1
                continue

            self.aligned_parts += 1
            marks = part_dynamics(whole_part)
            offset = 0
            for path in segments:
                try:
                    parts = ET.parse(path).getroot().findall("part")  # noqa: S314
                except ET.ParseError:
                    continue
                if part_index >= len(parts):
                    continue
                length = len(part_signature(parts[part_index]))
                page, system = (int(field) for field in path.stem.split(":")[1:])
                self.slices[(page, system, part_index)] = marks[offset : offset + length]
                offset += length

    def for_segment(self, page: int, system: int, part_index: int) -> list[DynamicMark] | None:
        return self.slices.get((page, system, part_index))


def apply_dynamics(part: ET.Element, marks: list[DynamicMark]) -> int:
    """Write dynamics onto a single part's notes, in place; returns how many landed.

    Inserted as ordinary `<direction><direction-type><dynamics>` elements immediately
    before the note they mark - the same convention MuseScore's own engraving follows,
    and the same rule `dynamics_attachment.py`/`NotationExtractor.handle_direction`
    already read for the "next real note in document order" attachment - so the ordinary
    extractor picks it up exactly the way it would a native direction, and nothing
    downstream needs to know this was reconstructed.

    Visible notes only, matching how the alignment was established.
    """
    if part.tag != "part":
        raise ValueError(f"expected a <part> element, got <{part.tag}>")

    applied = 0
    # Collected first, applied back to front: inserting a <direction> before note i would
    # otherwise shift the index of every note before it that still needs its own insert.
    pairs: list[tuple[ET.Element, ET.Element, DynamicMark]] = []
    mark_iter = iter(marks)
    for measure in part.findall("measure"):
        for note in measure.findall("note"):
            if not is_visible(note):
                continue
            mark = next(mark_iter, DynamicMark.NONE)
            if mark != DynamicMark.NONE:
                pairs.append((measure, note, mark))

    for measure, note, mark in reversed(pairs):
        index = list(measure).index(note)
        direction = ET.Element("direction")
        direction_type = ET.SubElement(direction, "direction-type")
        dynamics = ET.SubElement(direction_type, "dynamics")
        ET.SubElement(dynamics, mark.value)
        measure.insert(index, direction)
        applied += 1
    return applied
