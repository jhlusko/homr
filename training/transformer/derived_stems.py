"""
Derive stem direction from predicted beams instead of predicting it.

27.26 measured the stem head at 94.3% against 91.4% for a pitch rule the labels already
imply and 95.7% for a rule that also knows where the beam groups are. The head therefore
sits *below* the better rule - but that rule needed beam grouping, which homr did not have.

It does now: the beam heads predict grouping at 0.923 exact-vector accuracy. So this asks
whether feeding predicted beams into the rule beats the head that was trained for the job.
If it does, the stem head can go, and stem direction costs no parameters at all.

Four numbers are produced over the same notes, which is the only way the comparison means
anything:

  pitch alone            what a label file already implies
  predicted beams        the rule, grouped by what the beam heads predicted
  reference beams        the same rule with perfect grouping - an upper bound
  the trained head       reported by the evaluation, quoted here for comparison

Grouping is reconstructed defensively because predicted vectors need not be well formed: a
BEGIN with no END, or a FLAG inside a run, is a thing a head can emit and a rule must
survive.
"""

# flake8: noqa: T201

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from homr.transformer.structured_notation import BeamLevelState, StemDirection
from training.omr_datasets.stem_baseline import (
    _CLEF_LINE,
    _predict,
    middle_line,
    note_position,
    stated_stem,
)
from training.transformer.rule_vs_head import DEFAULT_TIME, _ordering, segment_for


@dataclass
class Note:
    position: int
    actual: StemDirection
    #: Index into the staff's beamable notes, or None where the note carries no flags.
    beamable: int | None


def groups_from_beams(vectors: list[tuple[BeamLevelState, ...]]) -> list[list[int]]:
    """Split note indices into beam groups using level 1 only.

    Level 1 carries the outer beam, which is what sets stem direction; deeper levels
    subdivide a group that has already chosen its direction.

    Written to survive vectors a head can emit but an engraver would not. A BEGIN while a
    group is open closes the previous one rather than nesting, and a FLAG or an
    inapplicable level ends the run - a flagged note cannot be inside a beam.
    """
    groups: list[list[int]] = []
    current: list[int] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for index, vector in enumerate(vectors):
        state = vector[0] if vector else BeamLevelState.NOT_APPLICABLE
        if state == BeamLevelState.BEGIN:
            flush()
            current = [index]
        elif state in (BeamLevelState.CONTINUE, BeamLevelState.END):
            if current:
                current.append(index)
            else:
                # A continuation with nothing open: the group began before this staff, or
                # the head emitted a state it could not support. Either way it stands
                # alone rather than joining whatever came before.
                groups.append([index])
            if state == BeamLevelState.END:
                flush()
        else:
            flush()
            groups.append([index])
    flush()
    return groups


def derive(notes: list[Note], vectors: list[tuple[BeamLevelState, ...]]) -> list[StemDirection]:
    """One direction per note, taken per beam group from the notehead furthest out."""
    directions: list[StemDirection] = [StemDirection.UNKNOWN] * len(notes)
    for group in groups_from_beams(vectors):
        extreme = max((notes[i].position for i in group), key=abs)
        chosen = _predict(extreme)
        for index in group:
            directions[index] = chosen
    return directions


def walk_part(part: ET.Element, levels: int) -> tuple[list[Note], list[bool]]:
    """Notes with a stated stem and a pitch, and which of them carry flags.

    Chord members are kept: they share a stem with their leader, so they share its group,
    and `stem_baseline` scores them too - excluding them here would make the numbers
    incomparable.
    """
    from homr.transformer.structured_notation import applicable_beam_levels

    middle = middle_line("G", _CLEF_LINE["G"])
    notes: list[Note] = []
    beamable: list[bool] = []

    for measure in part.findall("measure"):
        clef = measure.find("attributes/clef")
        if clef is not None:
            sign = clef.findtext("sign") or "G"
            line_text = clef.findtext("line")
            line = (
                int(line_text)
                if line_text and line_text.strip().isdigit()
                else _CLEF_LINE.get(sign, 3)
            )
            middle = middle_line(sign, line)

        for note in measure.findall("note"):
            actual = stated_stem(note)
            position = note_position(note, middle)
            if actual is None or position is None:
                continue
            flagged = (
                applicable_beam_levels(note.findtext("type")) > 0
                and note.find("rest") is None
            )
            notes.append(Note(position, actual, len(beamable) if flagged else None))
            if flagged:
                beamable.append(True)
    return notes, beamable


@dataclass
class Comparison:
    pitch_only: int = 0
    from_predicted: int = 0
    from_reference: int = 0
    total: int = 0
    skipped: int = 0

    def describe(self) -> str:
        def rate(value: int) -> str:
            return f"{value / self.total:.1%}" if self.total else "n/a"

        return "\n".join(
            [
                f"{self.total:,} notes with a stated stem"
                + (f", {self.skipped:,} staves skipped" if self.skipped else ""),
                "",
                f"  pitch alone                     {rate(self.pitch_only)}",
                f"  grouped by predicted beams      {rate(self.from_predicted)}",
                f"  grouped by reference beams      {rate(self.from_reference)}",
            ]
        )


def _vectors(record: dict, key: str, notes: list[Note], levels: int) -> list[tuple]:
    """Beam vectors laid back out over every note, not just the beamable ones."""
    supplied = record[key]
    out: list[tuple] = []
    for note in notes:
        if note.beamable is None or note.beamable >= len(supplied):
            out.append((BeamLevelState.NOT_APPLICABLE,) * levels)
        else:
            out.append(tuple(BeamLevelState(s) for s in supplied[note.beamable][:levels]))
    return out


def compare(predictions: Path, dataset_root: Path, levels: int) -> Comparison:
    result = Comparison()
    records = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in sorted(records, key=_ordering):
        located = segment_for(Path(record["tokens"]), dataset_root)
        if located is None:
            result.skipped += 1
            continue
        segment_path, part_index = located
        try:
            parts = ET.parse(segment_path).getroot().findall("part")  # noqa: S314
        except ET.ParseError:
            result.skipped += 1
            continue
        if part_index >= len(parts):
            result.skipped += 1
            continue

        notes, beamable = walk_part(parts[part_index], levels)
        if len(beamable) != len(record["reference"]):
            # The same alignment rule_vs_head enforces: a disagreement means the two walks
            # did not see the same notes, so nothing derived from the join can be trusted.
            result.skipped += 1
            continue

        predicted = derive(notes, _vectors(record, "predicted", notes, levels))
        reference = derive(notes, _vectors(record, "reference", notes, levels))
        for index, note in enumerate(notes):
            result.total += 1
            result.pitch_only += _predict(note.position) == note.actual
            result.from_predicted += predicted[index] == note.actual
            result.from_reference += reference[index] == note.actual
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    print(compare(args.predictions, args.dataset_root, args.levels).describe())


if __name__ == "__main__":
    main()
