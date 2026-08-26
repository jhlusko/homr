"""Recover barlines and repeats from the whole score, the way slurs and dynamics were.

Human review of the OSSQ pairs kept reporting the same thing: the notes were right and
the *repeats* were missing - "correct but missing final repeat", "missing a double repeat
between measures 3 and 4", "missing repeat in second last bar and double bar at end".

Measured across the 92 scores that have both a whole-score MusicXML and scanned
segments: the whole scores carry **8,617 `<barline>` elements, 3,740 of them repeats**,
and the segments carry **zero**. Not a few, none - the same total loss §28.1 recorded for
`<direction>` and §27.20 for slur placement, from the same cause: the MuseScore round
trip that produces the segments drops them.

This is not a vocabulary limitation. `barline`, `doublebarline`, `bolddoublebarline`,
`repeatStart`, `repeatEnd` and `repeatEndStart` all exist in the token vocabulary, so a
recovered barline tokenises without any change to the model.

**Why the join is by measure, not by note.** `slur_placement` and `dynamics_placement`
attach to notes and align on a note signature. A barline belongs to a *measure*, so the
alignment has to hold at measure granularity too - which is an additional claim, and one
worth checking rather than assuming, given how many alignment assumptions in this corpus
have turned out to be wrong. Checked over the parts whose note signatures already match:
**64 of 64 also have identical measure counts, zero disagreements**. So a part whose
notes align has measures that align, and slicing the whole part's per-measure barlines by
each segment's measure count is sound.

The note-level alignment is still the gate. A part whose notes do not concatenate is
skipped entirely rather than being joined on measures alone: matching measure counts
would be far weaker evidence on its own, and a wrong repeat is worse than a missing one -
it changes how the music is *played*, not merely how it looks.
"""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.slur_placement import concatenated, part_signature, segments_of


def part_barlines(whole_part: ET.Element) -> list[list[ET.Element]]:
    """Per measure, the barline elements it carries (deep-copied, in document order)."""
    return [
        [copy.deepcopy(barline) for barline in measure.findall("barline")]
        for measure in whole_part.findall("measure")
    ]


class BarlinePlacementIndex:
    """Barlines for every segment of one score, or nothing if it cannot be trusted.

    Structurally the same as `DynamicsPlacementIndex`: built per score and sliced by
    segment rather than consumed as a running cursor, because a system skipped for a crop
    mismatch still occupies measures in the whole score and a cursor would misalign after
    the first skip.
    """

    def __init__(self, work: Path, score_id: str, whole: Path) -> None:
        self.slices: dict[tuple[int, int, int], list[list[ET.Element]]] = {}
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

            per_measure = part_barlines(whole_part)
            offset = 0
            usable = True
            plan: dict[tuple[int, int, int], list[list[ET.Element]]] = {}
            for path in segments:
                try:
                    parts = ET.parse(path).getroot().findall("part")  # noqa: S314
                except ET.ParseError:
                    continue
                if part_index >= len(parts):
                    continue
                count = len(parts[part_index].findall("measure"))
                if offset + count > len(per_measure):
                    # Measure counts disagree for this part after all; take nothing from
                    # it rather than aligning part of it and guessing the rest.
                    usable = False
                    break
                page, system = (int(field) for field in path.stem.split(":")[1:])
                plan[(page, system, part_index)] = per_measure[offset : offset + count]
                offset += count

            if usable:
                self.aligned_parts += 1
                self.slices.update(plan)
            else:
                self.skipped_parts += 1

    def for_segment(
        self, page: int, system: int, part_index: int
    ) -> list[list[ET.Element]] | None:
        return self.slices.get((page, system, part_index))


def apply_barlines(part: ET.Element, per_measure: list[list[ET.Element]]) -> int:
    """Write barlines back onto a part's measures, in place; returns how many landed.

    Placement follows MusicXML's own convention rather than the original index: a
    `location="left"` barline belongs at the head of its measure (after `<attributes>`,
    which must come first), and everything else at the end. Reconstructing the exact
    original offset would mean tracking note positions for no benefit - a barline's
    meaning comes from its `location`, not from how many notes precede it in the file.
    """
    if part.tag != "part":
        raise ValueError(f"expected a <part> element, got <{part.tag}>")

    applied = 0
    measures = part.findall("measure")
    for measure, barlines in zip(measures, per_measure, strict=False):
        for barline in barlines:
            if barline.get("location") == "left":
                attributes = measure.find("attributes")
                index = list(measure).index(attributes) + 1 if attributes is not None else 0
                measure.insert(index, copy.deepcopy(barline))
            else:
                measure.append(copy.deepcopy(barline))
            applied += 1
    return applied
