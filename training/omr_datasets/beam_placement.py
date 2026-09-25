"""Recover rest-spanning-beam ground truth (MuseScore's `<Rest><BeamMode>`) and join it
to scanned-track crops, the way slur placement, dynamics and barlines were recovered.

A3 (`OTS_HOMR_RELEASE_SEQUENCE_2026-09-25.md`) needs to know, for each rest in a scanned
crop, whether the source score beamed it into the surrounding run - `evaluate_structured_
heads.py` drops rests from `predictions.jsonl` entirely (every rest's `beam_levels` are
`NOT_APPLICABLE`), and MusicXML has no `<beam>` concept for rests at all, so this
information exists nowhere but the MuseScore `.mscx` source.

**Two joins, not one, and they use different keys.** `slur_placement.py`'s
`PlacementIndex` already solves "segment note *k* of a part is whole-score note *k* of
that part" - a positional join between two MusicXML documents, gated on matching note
*signatures* end to end. That join is reused here unchanged (via `segments_of(...,
track="scanned")`, `concatenated`, `part_signature`). It does not reach the `.mscx`,
which has no `<note>` elements to compare signatures against.

The second join - whole-score MusicXML position to `.mscx` position - is new, and is
**not** a plain note-count join. A first attempt joining raw `<note>` position to raw
`<Chord>`/`<Rest>` position diverged after the first chord in a real file: MusicXML gives
every pitch of a chord its own `<note>` element, while `.mscx` gives the whole chord one
`<Chord>` element. Collapsing MusicXML chord-continuation notes (`<chord/>` present)
before counting fixes it - checked on all rest positions, not just counts, across every
part of one real score: 3 of 4 parts align exactly end to end, one does not and is
excluded, matching the same per-part gate `PlacementIndex`/`BarlinePlacementIndex`
already use rather than trusting an unverified part.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.slur_placement import concatenated, is_visible, part_signature, segments_of

#: MuseScore's own vocabulary for "this rest sits inside a beamed run" - matches
#: `results/beam_mode_rest_adjacency_check.json`'s already-validated population (98.6%
#: of these have both neighbours also beamed).
NONTRIVIAL_BEAM_MODES = frozenset({"begin", "mid", "end", "begin32", "begin64"})

_CHORD_OR_REST = re.compile(r"<(Chord|Rest)\b")
_BEAM_MODE = re.compile(r"<BeamMode>([^<]*)</BeamMode>")


def mscx_content_staves(mscx_text: str) -> list[str]:
    """The `.mscx` main score's content staves, in `<Part>` order.

    A `.mscx` file names `<Staff id="N">` twice per part: once under `<Part>` for
    instrument metadata (no `<Measure>` children), once at `<Score>` level for the
    actual notes. Only the second kind is usable here, and only the first four belong
    to the main condensed score - a `.mscx` bundles one further `<Score>` block per
    instrument as a duplicate solo-part excerpt (confirmed by inspection: a 4-part
    quartet's file has 5 `<Score>` blocks), which would silently double-count staves if
    not excluded.
    """
    main = re.search(r"<Score>.*?</Score>", mscx_text, re.S)
    if main is None:
        return []
    blocks = re.findall(r'<Staff id="\d+">(.*?)</Staff>', main.group(0), re.S)
    return [content for content in blocks if "<Measure" in content]


def mscx_rests_by_staff(mscx_text: str) -> list[list[tuple[bool, str | None]]]:
    """Per content staff, `(is_rest, beam_mode)` for every `<Chord>`/`<Rest>`, in order.

    `beam_mode` is `None` for a `<Chord>` (beams are never recorded on notes this way in
    `.mscx` - only a rest can carry an explicit "include me in the beam" marker) and for
    a `<Rest>` with no `<BeamMode>` tag or a trivial one (not in `NONTRIVIAL_BEAM_MODES`).
    """
    result = []
    for staff in mscx_content_staves(mscx_text):
        elements = []
        for match in _CHORD_OR_REST.finditer(staff):
            is_rest = match.group(1) == "Rest"
            beam_mode = None
            if is_rest:
                # BeamMode, if present, is inside this element - search a bounded
                # window rather than the whole remaining document.
                window = staff[match.end() : match.end() + 400]
                close = window.find("</Rest>")
                window = window[: close if close != -1 else len(window)]
                found = _BEAM_MODE.search(window)
                if found and found.group(1) in NONTRIVIAL_BEAM_MODES:
                    beam_mode = found.group(1)
            elements.append((is_rest, beam_mode))
        result.append(elements)
    return result


def collapsed_rest_flags(part: ET.Element) -> tuple[list[bool], list[int]]:
    """This part's notes, MusicXML-chord-collapsed to match `.mscx`'s one-element-per-
    chord shape, as `(is_rest per collapsed position, collapsed index per uncollapsed
    position)`.

    The second list is the bridge back to `concatenated()`'s uncollapsed, segment-
    joined positions: a rest is never a chord member, so every rest's uncollapsed index
    always has a defined collapsed index.
    """
    notes = [
        note
        for measure in part.findall("measure")
        for note in measure.findall("note")
        if is_visible(note)
    ]
    is_rest_collapsed: list[bool] = []
    collapsed_of: list[int] = []
    for note in notes:
        if note.find("chord") is not None:
            # Shares the preceding collapsed slot; never itself a rest.
            collapsed_of.append(len(is_rest_collapsed) - 1)
            continue
        collapsed_of.append(len(is_rest_collapsed))
        is_rest_collapsed.append(note.find("rest") is not None)
    return is_rest_collapsed, collapsed_of


class BeamPlacementIndex:
    """Rest `BeamMode` for every scanned-track segment of one score, or nothing for a
    part that cannot be trusted end to end.

    Mirrors `PlacementIndex`'s shape (built once per score, sliced by segment, gated per
    part) but joins through two documents - whole-score MusicXML to `.mscx`, and
    whole-score MusicXML to each segment - rather than one.
    """

    def __init__(self, work: Path, score_id: str, whole: Path, mscx_path: Path) -> None:
        self.slices: dict[tuple[int, int, int], list[tuple[bool, str | None]]] = {}
        self.aligned_parts = 0
        self.skipped_parts = 0
        self._build(work, score_id, whole, mscx_path)

    def _build(self, work: Path, score_id: str, whole: Path, mscx_path: Path) -> None:
        try:
            whole_parts = ET.parse(whole).getroot().findall("part")  # noqa: S314
        except (ET.ParseError, OSError):
            return
        segments = segments_of(work, score_id, track="scanned")
        if not segments:
            return
        try:
            mscx_text = mscx_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return
        mscx_staves = mscx_rests_by_staff(mscx_text)

        for part_index, whole_part in enumerate(whole_parts):
            if part_index >= len(mscx_staves):
                self.skipped_parts += 1
                continue

            # Gate 1: this part's segments reconstruct the whole part, note for note
            # (the join `PlacementIndex` already validated).
            expected = part_signature(whole_part)
            found, _ = concatenated(segments, part_index)
            if len(expected) != len(found) or expected != found:
                self.skipped_parts += 1
                continue

            # Gate 2: the whole part's collapsed rest/chord pattern matches the .mscx
            # staff's, position for position - not just in count.
            is_rest_collapsed, collapsed_of = collapsed_rest_flags(whole_part)
            mscx_staff = mscx_staves[part_index]
            mscx_is_rest = [is_rest for is_rest, _ in mscx_staff]
            if is_rest_collapsed != mscx_is_rest:
                self.skipped_parts += 1
                continue

            self.aligned_parts += 1
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
                slice_rests = []
                for uncollapsed_index in range(offset, offset + length):
                    if not expected[uncollapsed_index][4]:  # not a rest
                        continue
                    collapsed_index = collapsed_of[uncollapsed_index]
                    slice_rests.append(mscx_staff[collapsed_index])
                self.slices[(page, system, part_index)] = slice_rests
                offset += length

    def for_segment(self, page: int, system: int, part_index: int) -> list[tuple[bool, str | None]] | None:
        return self.slices.get((page, system, part_index))
