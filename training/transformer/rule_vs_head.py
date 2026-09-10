"""
Is the head right where the rule is wrong?

This is the question Gate C turns on, and neither the baseline nor the evaluation answers
it alone. A head that scored exactly the baseline's 84% might have learned the rule and
nothing else - identical accuracy, zero added value - or it might be right on a different
84%. Only the crosstab distinguishes them:

```
                    head right   head wrong
  rule right              A            B
  rule wrong              C            D
```

`C` is the whole point of the exercise: beaming the head recovered that duration and
metre cannot predict, which is by definition something only the image carries. `B` is its
price - places the rule already had right and the head lost.

The join is by note, and notes are the thing this pipeline keeps getting wrong (27.11,
27.15, 27.17), so it is checked rather than assumed: the rule is walked over the same
MusicXML part the labels came from, and an example whose note count disagrees with its
prediction record is skipped and counted, never truncated into alignment.
"""

# flake8: noqa: T201

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.structured_notation import BeamLevelState, applicable_beam_levels

DEFAULT_TIME = (4, 4)


@dataclass
class Crosstab:
    rule_right_head_right: int = 0
    rule_right_head_wrong: int = 0
    rule_wrong_head_right: int = 0
    rule_wrong_head_wrong: int = 0
    skipped_examples: int = 0
    joined_examples: int = 0
    chord_members_skipped: int = 0

    @property
    def notes(self) -> int:
        return (
            self.rule_right_head_right
            + self.rule_right_head_wrong
            + self.rule_wrong_head_right
            + self.rule_wrong_head_wrong
        )

    @property
    def rule_accuracy(self) -> float:
        right = self.rule_right_head_right + self.rule_right_head_wrong
        return right / self.notes if self.notes else 0.0

    @property
    def head_accuracy(self) -> float:
        right = self.rule_right_head_right + self.rule_wrong_head_right
        return right / self.notes if self.notes else 0.0

    @property
    def exceptions_recovered(self) -> float:
        """Of the notes the rule gets wrong, the share the head gets right."""
        wrong = self.rule_wrong_head_right + self.rule_wrong_head_wrong
        return self.rule_wrong_head_right / wrong if wrong else 0.0

    @property
    def agreements_lost(self) -> float:
        """Of the notes the rule gets right, the share the head gets wrong."""
        right = self.rule_right_head_right + self.rule_right_head_wrong
        return self.rule_right_head_wrong / right if right else 0.0

    def observe(self, rule_right: bool, head_right: bool) -> None:
        if rule_right and head_right:
            self.rule_right_head_right += 1
        elif rule_right:
            self.rule_right_head_wrong += 1
        elif head_right:
            self.rule_wrong_head_right += 1
        else:
            self.rule_wrong_head_wrong += 1

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.joined_examples:,} staves joined, {self.notes:,} beamable notes"
                + (f", {self.skipped_examples:,} skipped" if self.skipped_examples else ""),
                "",
                "                  head right   head wrong",
                f"  rule right   {self.rule_right_head_right:>11,}"
                f"{self.rule_right_head_wrong:>13,}",
                f"  rule wrong   {self.rule_wrong_head_right:>11,}"
                f"{self.rule_wrong_head_wrong:>13,}",
                "",
                f"rule accuracy on these notes: {self.rule_accuracy:.1%}",
                f"head accuracy on these notes: {self.head_accuracy:.1%}",
                f"exceptions the head recovers: {self.exceptions_recovered:.1%}"
                f"  ({self.rule_wrong_head_right:,} notes)",
                f"agreements the head loses:    {self.agreements_lost:.1%}"
                f"  ({self.rule_right_head_wrong:,} notes)",
            ]
        )


def segment_for(tokens: Path, dataset_root: Path) -> tuple[Path, int] | None:
    """The MusicXML segment and part index a token file came from.

    convert_ossq names token files `{score}_{page}_{system}_{part}.txt`, with the part
    1-based, so the segment is recoverable without a side table. Returns None when the
    name does not decompose or the segment is missing, which the caller counts rather
    than guessing at.
    """
    fields = tokens.stem.rsplit("_", 3)
    if len(fields) != 4:
        return None
    score, page, system, part = fields
    if not (page.isdigit() and system.isdigit() and part.isdigit()):
        return None
    name = f"{score}:{page}:{system}.musicxml"
    for candidate in dataset_root.glob(f"scores/*/*/musicxml/unaligned/{name}"):
        return candidate, int(part) - 1
    return None


Meter = tuple[int, int, int]


def rule_vectors(
    part: ET.Element, meter: Meter = (1, *DEFAULT_TIME)
) -> tuple[list[tuple[tuple[BeamLevelState, ...], bool]], Meter]:
    """The rule's beam vector for every note of one part, with whether it is a chord member.

    Chord members are carried so this lines up with the labels, which have one entry per
    <note>, but they are flagged so the caller can decline to score them. They have to be
    excluded, for a reason that is about markup rather than music: MusicXML writes <beam>
    only on a chord's first note - 1,703 chord members in a 600-segment sample, not one of
    them carrying a beam element - so the extractor labels every one of them FLAG while
    the rule repeats the leader's BEGIN or END. Scoring them would manufacture a
    disagreement on roughly one flagged note in twenty that is a notation convention, not
    an engraving exception, and `beam_baseline.py` already counts one decision per stem
    for the same reason.
    """
    vectors: list[tuple[tuple[BeamLevelState, ...], bool]] = []
    divisions, beats, beat_type = meter

    for measure in part.findall("measure"):
        divisions_text = measure.findtext("attributes/divisions")
        if divisions_text and divisions_text.strip().isdigit():
            divisions = int(divisions_text)
        time = measure.find("attributes/time")
        if time is not None:
            beats_text, type_text = time.findtext("beats"), time.findtext("beat-type")
            if beats_text and type_text and beats_text.isdigit() and type_text.isdigit():
                beats, beat_type = int(beats_text), int(type_text)

        onsets: dict[str, int] = {}
        by_voice: dict[str, list[BeamableNote]] = {}
        # Where each note element's vector belongs, which chord leader it follows, and
        # whether it is itself a chord member.
        slots: list[tuple[str, int, bool]] = []
        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            if note.find("chord") is not None and by_voice.get(voice):
                slots.append((voice, len(by_voice[voice]) - 1, True))
                continue
            duration_text = note.findtext("duration")
            duration = int(duration_text) if duration_text and duration_text.isdigit() else 0
            onset = onsets.get(voice, 0)
            by_voice.setdefault(voice, []).append(
                BeamableNote(
                    onset=onset,
                    duration=duration,
                    flags=applicable_beam_levels(note.findtext("type")),
                    is_rest=note.find("rest") is not None,
                )
            )
            slots.append((voice, len(by_voice[voice]) - 1, False))
            onsets[voice] = onset + duration

        beat = beat_divisions(beats, beat_type, divisions)
        wide = wide_unit(beats, beat_type, divisions)
        computed = {
            voice: automatic_beams(notes, beat, wide) for voice, notes in by_voice.items()
        }
        for voice, index, is_chord_member in slots:
            vectors.append((computed[voice][index], is_chord_member))
    return vectors, (divisions, beats, beat_type)


def _ordering(record: dict) -> tuple[str, int, int, int]:
    """Score, page, system, part - the order a movement is actually read in."""
    fields = Path(record["tokens"]).stem.rsplit("_", 3)
    if len(fields) != 4 or not all(f.isdigit() for f in fields[1:]):
        return ("", 0, 0, 0)
    return (fields[0], int(fields[1]), int(fields[2]), int(fields[3]))


def compare(predictions: Path, dataset_root: Path, levels: int) -> Crosstab:
    """Score every staff, carrying the meter across the segments of each part.

    The meter has to be carried because a systemwise segment restates <time> only at a
    movement start or a genuine change - so a segment taken alone is beamed as if it were
    in 4/4. That is not a small effect: it dropped the rule's measured accuracy from
    87.0% to 83.5% against the same split's baseline, and every point it loses is a point
    wrongly credited to the head as an exception recovered.

    Records are sorted rather than trusted to arrive in order, since the carry is only
    correct if a part's segments are seen in reading order.
    """
    crosstab = Crosstab()
    meters: dict[tuple[str, int], Meter] = {}
    records = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    for record in sorted(records, key=_ordering):
        located = segment_for(Path(record["tokens"]), dataset_root)
        if located is None:
            crosstab.skipped_examples += 1
            continue
        segment_path, part_index = located
        score_id = _ordering(record)[0]
        try:
            parts = ET.parse(segment_path).getroot().findall("part")  # noqa: S314
        except ET.ParseError:
            crosstab.skipped_examples += 1
            continue
        if part_index >= len(parts):
            crosstab.skipped_examples += 1
            continue

        carried = meters.get((score_id, part_index), (1, *DEFAULT_TIME))
        rules, carried = rule_vectors(parts[part_index], carried)
        meters[(score_id, part_index)] = carried
        beamable = [
            (vector, is_chord_member)
            for vector, is_chord_member in rules
            if any(state != BeamLevelState.NOT_APPLICABLE for state in vector[:levels])
        ]
        reference = record["reference"]
        if len(beamable) != len(reference):
            # The join is by position, and this pipeline has produced three separate
            # position bugs already. A length disagreement means the two walks did not
            # see the same notes, so nothing here can be trusted - skip it whole.
            crosstab.skipped_examples += 1
            continue

        crosstab.joined_examples += 1
        for (rule, is_chord_member), truth, head in zip(
            beamable, reference, record["predicted"], strict=True
        ):
            if is_chord_member:
                # Aligned but not scored - see rule_vectors.
                crosstab.chord_members_skipped += 1
                continue
            engraved = tuple(truth[:levels])
            crosstab.observe(
                tuple(str(s) for s in rule[:levels]) == engraved,
                tuple(head[:levels]) == engraved,
            )
    return crosstab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    print(compare(args.predictions, args.dataset_root, args.levels).describe())


if __name__ == "__main__":
    main()
