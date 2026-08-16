"""
How much of stem direction is already implicit in a label file?

The beam heads have a baseline to clear (`beam_baseline.py`, 82% corpus-wide), and the
stem head has none - 27.21 reports macro F1 0.719 against nothing at all. That is the
wrong way round: stem direction is the *most* rule-governed of the three notations, so a
head that cannot beat the textbook rule has learned nothing worth keeping.

The rule engravers use, and the one implemented here:

  - a note at or above the middle line takes a down stem, below it an up stem;
  - a chord takes the direction of whichever notehead is furthest from the middle line;
  - where two voices share a staff the convention overrides pitch entirely - upper voice
    up, lower voice down - which is why voice is checked first.

Three variants are reported, because they need different amounts of information and only
the first is a fair statement of what a label file already implies:

  pitch alone           what a token file carries - rhythm, pitch, clef, position. Voice
                        is *not* in it: symbols are ordered by position within a measure
                        and never say which voice they belong to.
  pitch and voice       needs the MusicXML, and turns out to be worth 0.3 points, because
                        quartet parts are overwhelmingly single-voice.
  and beam grouping     an upper bound rather than a rule: real engraving sets one
                        direction per beam group, not per note, so the per-note rule is
                        charged for every beamed run that crosses the middle line. Using
                        the engraved beams to fix the groups is information the other two
                        do not have.
"""

# flake8: noqa: T201

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from homr.transformer.structured_notation import StemDirection
from training.omr_datasets.ossq_splits import load_split_manifest

#: Diatonic index of each step within an octave.
_STEPS = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}

#: The pitch a clef sign fixes, as (step, octave), before its line is taken into account.
_CLEF_PITCH = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}

#: Where the sign sits by default, so a clef with no <line> still resolves.
_CLEF_LINE = {"G": 2, "F": 4, "C": 3}


def diatonic(step: str, octave: int) -> int:
    return octave * 7 + _STEPS.get(step.strip().upper(), 0)


def middle_line(sign: str, line: int) -> int:
    """The diatonic index sitting on the middle line of the staff.

    A clef fixes one pitch to one line; the middle line is line 3, and each line is two
    diatonic steps from the next. So G on line 2 puts B4 in the middle, F on line 4 puts
    D3 there, and C on line 3 puts C4 there - the three clefs a quartet uses.
    """
    step, octave = _CLEF_PITCH.get(sign.strip().upper(), ("G", 4))
    return diatonic(step, octave) + 2 * (3 - line)


def note_position(note: ET.Element, middle: int) -> int | None:
    """Diatonic steps above the middle line, or None for a note with no pitch."""
    pitch = note.find("pitch")
    if pitch is None:
        return None
    step = pitch.findtext("step")
    octave = pitch.findtext("octave")
    if not step or not octave or not octave.strip().lstrip("-").isdigit():
        return None
    return diatonic(step, int(octave)) - middle


def stated_stem(note: ET.Element) -> StemDirection | None:
    text = (note.findtext("stem") or "").strip().lower()
    if text == "up":
        return StemDirection.UP
    if text == "down":
        return StemDirection.DOWN
    return None


@dataclass
class StemBaseline:
    matching: int = 0
    total: int = 0
    by_actual: Counter[str] = field(default_factory=Counter)
    wrong_by_actual: Counter[str] = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        return self.matching / self.total if self.total else 0.0

    def observe(self, predicted: StemDirection, actual: StemDirection) -> None:
        self.total += 1
        self.by_actual[str(actual)] += 1
        if predicted == actual:
            self.matching += 1
        else:
            self.wrong_by_actual[str(actual)] += 1

    def describe(self, label: str) -> str:
        lines = [
            f"{label}: {self.matching:,} / {self.total:,} = {self.rate:.1%}",
        ]
        for name, count in self.by_actual.most_common():
            wrong = self.wrong_by_actual[name]
            lines.append(f"    {name}: {count - wrong:,}/{count:,} = {(count - wrong) / count:.1%}")
        return "\n".join(lines)


def _predict(position: int) -> StemDirection:
    """A note on or above the middle line takes a down stem."""
    return StemDirection.DOWN if position >= 0 else StemDirection.UP


def measure_part(
    part: ET.Element,
    strict: StemBaseline,
    grouped: StemBaseline,
    pitch_only: StemBaseline | None = None,
) -> None:
    middle = middle_line("G", _CLEF_LINE["G"])

    for measure in part.findall("measure"):
        clef = measure.find("attributes/clef")
        if clef is not None:
            sign = clef.findtext("sign") or "G"
            line_text = clef.findtext("line")
            line = int(line_text) if line_text and line_text.strip().isdigit() else _CLEF_LINE.get(sign, 3)
            middle = middle_line(sign, line)

        voices = {note.findtext("voice") or "1" for note in measure.findall("note")}
        multi_voice = len(voices) > 1

        # (note, position, actual) for every note that states a stem and has a pitch,
        # split into the runs the engraved beams group together.
        run: list[tuple[int, StemDirection]] = []
        runs: list[list[tuple[int, StemDirection]]] = []
        for note in measure.findall("note"):
            actual = stated_stem(note)
            position = note_position(note, middle)
            if actual is None or position is None:
                continue
            voice = note.findtext("voice") or "1"

            if multi_voice:
                predicted = StemDirection.UP if voice == "1" else StemDirection.DOWN
            else:
                predicted = _predict(position)
            strict.observe(predicted, actual)
            if pitch_only is not None:
                # No voice: the token format orders symbols by position within a measure
                # and never says which voice a symbol belongs to, so this is what is
                # actually derivable from a label file rather than from MusicXML.
                pitch_only.observe(_predict(position), actual)

            beams = [(b.text or "").strip() for b in note.findall("beam")]
            run.append((position, actual))
            if not beams or "end" in beams:
                runs.append(run)
                run = []
        if run:
            runs.append(run)

        for group in runs:
            if not group:
                continue
            # One direction for the whole group, set by the notehead furthest from the
            # middle line - which is what an engraver does.
            extreme = max(group, key=lambda entry: abs(entry[0]))[0]
            predicted = _predict(extreme)
            for _, actual in group:
                grouped.observe(predicted, actual)


def measure(paths: list[Path]) -> tuple[StemBaseline, StemBaseline, StemBaseline]:
    strict, grouped, pitch_only = StemBaseline(), StemBaseline(), StemBaseline()
    for path in paths:
        try:
            root = ET.parse(path).getroot()  # noqa: S314
        except ET.ParseError:
            continue
        for part in root.findall("part"):
            measure_part(part, strict, grouped, pitch_only)
    return strict, grouped, pitch_only


def score_files(dataset_root: Path, track: str, split: str | None) -> list[Path]:
    manifest = load_split_manifest()
    chosen: list[Path] = []
    for directory in sorted(dataset_root.glob("scores/*/*")):
        for path in sorted(directory.glob("*.musicxml")):
            score_id = path.stem.removesuffix("_cleaned")
            if path.stem == score_id and (directory / f"{score_id}_cleaned.musicxml").exists():
                continue
            assigned = manifest.split_for(score_id, track)
            if assigned is None or (split is not None and assigned != split):
                continue
            chosen.append(path)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--track", choices=["synthetic", "scanned"], default="synthetic")
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = score_files(args.dataset_root, args.track, args.split)
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} score file(s), split={args.split or 'all'}")

    strict, grouped, pitch_only = measure(paths)
    print(pitch_only.describe("pitch alone, per note (what a label file carries)"))
    print(strict.describe("pitch and voice, per note (needs MusicXML)"))
    print(grouped.describe("pitch, voice and engraved beam grouping (upper bound)"))


if __name__ == "__main__":
    main()
