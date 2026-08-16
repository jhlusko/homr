"""
Can slur placement be recovered from the original scores, and is the join trustworthy?

27.20 found slur placement absent from every built label: the MuseScore round-trip that
produces the systemwise segments stores slur direction as automatic and omits it, so
`slur.slot.N.side` has no supervision at all. The information still exists, on half the
slurs in the original whole-score MusicXML, and the only way to it is a positional join -
segment note *k* of a part is whole-score note *k* of that part.

That is exactly the correspondence this pipeline has broken five separate times, so this
tool answers the precondition before anything is built on it: concatenate a part's
segments in reading order, compare against the same part of the whole score, and report
how often the two agree note for note.

Agreement is checked on a signature rather than a count. Two sequences of the same length
can still be different music - a dropped grace note and an added one cancel in a count and
not in a signature - and a wrong join here would silently attach one note's slur direction
to another.
"""

# flake8: noqa: T201

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from training.omr_datasets.ossq_splits import load_split_manifest

#: What identifies a note strongly enough to trust a positional join, without being so
#: strict that a harmless normalisation breaks it.
NoteSignature = tuple[str, str, str, bool, bool]


def note_signature(note: ET.Element) -> NoteSignature:
    pitch = note.find("pitch")
    step = pitch.findtext("step", "") if pitch is not None else ""
    octave = pitch.findtext("octave", "") if pitch is not None else ""
    return (
        (note.findtext("type") or "").strip(),
        step,
        octave,
        note.find("chord") is not None,
        note.find("rest") is not None,
    )


def is_visible(note: ET.Element) -> bool:
    """Whether this note is printed.

    Invisible notes are why the naive join fails. The segmentation drops most of them, so
    a whole-score walk that keeps them runs ahead of the segments and every placement
    after the first one lands on the wrong note. Measured across 16 parts, the shortfall
    equals the invisible count exactly - grace notes, the other obvious suspect, match to
    the note on every part - and excluding them makes the two walks agree precisely.
    """
    return note.get("print-object") != "no"


def part_signature(part: ET.Element) -> list[NoteSignature]:
    return [
        note_signature(note)
        for measure in part.findall("measure")
        for note in measure.findall("note")
        if is_visible(note)
    ]


def part_placements(part: ET.Element) -> list[dict[str, str]]:
    """Slur placement per note, as `{slur number: placement}`, in document order.

    Only slurs that state a placement appear. An empty dict is a note with no slur, or a
    slur whose direction the source leaves to the engraver - both mean "nothing to
    transfer" rather than "no slur".
    """
    placements: list[dict[str, str]] = []
    for measure in part.findall("measure"):
        for note in measure.findall("note"):
            if not is_visible(note):
                continue
            stated: dict[str, str] = {}
            for notations in note.findall("notations"):
                for slur in notations.findall("slur"):
                    placement = slur.get("placement") or slur.get("orientation")
                    if placement:
                        stated[slur.get("number") or "1"] = placement
            placements.append(stated)
    return placements


@dataclass
class Alignment:
    parts_checked: int = 0
    parts_aligned: int = 0
    length_mismatch: int = 0
    signature_mismatch: int = 0
    notes_aligned: int = 0
    placements_available: int = 0
    first_divergence: Counter[str] = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        return self.parts_aligned / self.parts_checked if self.parts_checked else 0.0

    def describe(self) -> str:
        lines = [
            f"parts checked: {self.parts_checked:,}",
            f"  aligned note for note: {self.parts_aligned:,} ({self.rate:.1%})",
            f"  length mismatch:       {self.length_mismatch:,}",
            f"  signature mismatch:    {self.signature_mismatch:,}",
            f"notes in aligned parts:  {self.notes_aligned:,}",
            f"slur placements recoverable on them: {self.placements_available:,}",
        ]
        if self.first_divergence:
            lines.append("")
            lines.append("where the signatures first differ (whole -> segments):")
            for pair, count in self.first_divergence.most_common(5):
                lines.append(f"  {pair}: {count:,}")
        return "\n".join(lines)


def segments_of(work: Path, score_id: str) -> list[Path]:
    """A score's systemwise segments in reading order."""
    return sorted(
        (work / "musicxml" / "unaligned").glob(f"{score_id}:*.musicxml"),
        key=lambda path: tuple(int(field) for field in path.stem.split(":")[1:]),
    )


def concatenated(segments: list[Path], part_index: int) -> tuple[list[NoteSignature], int]:
    """One part's notes across every segment, in reading order."""
    signatures: list[NoteSignature] = []
    seen = 0
    for path in segments:
        try:
            parts = ET.parse(path).getroot().findall("part")  # noqa: S314
        except ET.ParseError:
            continue
        if part_index >= len(parts):
            continue
        seen += 1
        signatures.extend(part_signature(parts[part_index]))
    return signatures, seen


def check_score(work: Path, score_id: str, whole: Path, alignment: Alignment) -> None:
    try:
        whole_parts = ET.parse(whole).getroot().findall("part")  # noqa: S314
    except ET.ParseError:
        return
    segments = segments_of(work, score_id)
    if not segments:
        return

    for part_index, whole_part in enumerate(whole_parts):
        alignment.parts_checked += 1
        expected = part_signature(whole_part)
        found, _ = concatenated(segments, part_index)

        if len(expected) != len(found):
            alignment.length_mismatch += 1
            continue
        divergence = next(
            ((a, b) for a, b in zip(expected, found, strict=True) if a != b), None
        )
        if divergence is not None:
            alignment.signature_mismatch += 1
            alignment.first_divergence[f"{divergence[0]} -> {divergence[1]}"] += 1
            continue

        alignment.parts_aligned += 1
        alignment.notes_aligned += len(found)
        alignment.placements_available += sum(1 for stated in part_placements(whole_part) if stated)


def measure(dataset_root: Path, track: str, split: str | None, limit: int = 0) -> Alignment:
    manifest = load_split_manifest()
    alignment = Alignment()
    checked = 0
    for work in sorted((dataset_root / "scores").glob("*/*")):
        if not work.is_dir():
            continue
        for whole in sorted(work.glob("*.musicxml")):
            score_id = whole.stem.removesuffix("_cleaned")
            # The original, not the cleaned copy: cleaning is what drops placement.
            if whole.stem != score_id:
                continue
            assigned = manifest.split_for(score_id, track)
            if assigned is None or (split is not None and assigned != split):
                continue
            check_score(work, score_id, whole, alignment)
            checked += 1
            if limit and checked >= limit:
                return alignment
    return alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--track", choices=["synthetic", "scanned"], default="synthetic")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N scores.")
    args = parser.parse_args()

    print(measure(args.dataset_root, args.track, args.split, args.limit).describe())


if __name__ == "__main__":
    main()
