"""
Measure how much of the corpus's beaming a fixed rule already reproduces.

This is the Gate C baseline. 15.2 asks whether the beam heads beat deterministic
reconstruction, so the number a head has to clear is not zero - it is whatever duration
and metre alone can predict, which 27.12 puts near four fifths.

That measurement was originally taken with a throwaway script, which meant the headline
figure could not be re-derived or narrowed to a split. This is the same measurement as a
committed tool, and it takes `--split`, so the baseline can be quoted on exactly the
scores a head is evaluated against rather than corpus-wide.

The comparison is per beam vector, over *notes* that carry at least one flag. Two
exclusions, both for the same reason - they are agreements the rule is not entitled to
claim credit for:

Notes with no flags. A quarter note carries no beam under any rule and none in any
engraving, so counting it would put the baseline in the high nineties.

Rests. An eighth rest has a flag count but no stem, so it can carry no beam: the rule
returns not-applicable and the engraving says the same, and every one of them would score
as a free match. They are 11.4% of what would otherwise be counted on the validation
split, and including them overstated this baseline by about 1.7 points.
"""

# flake8: noqa: T201

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.structured_notation import BeamLevelState, applicable_beam_levels
from training.omr_datasets.structured_notation_parser import NotationExtractor
from training.omr_datasets.ossq_splits import load_split_manifest

#: A part with no explicit time signature. 4/4 is the overwhelming default in this
#: corpus, and a part that never states one is measured rather than skipped - skipping it
#: would quietly drop music from the denominator.
DEFAULT_TIME = (4, 4)
DEFAULT_DIVISIONS = 1


@dataclass
class Baseline:
    matching: int = 0
    total: int = 0
    disagreements: Counter[tuple[str, str]] = field(default_factory=Counter)

    @property
    def rate(self) -> float:
        return self.matching / self.total if self.total else 0.0

    def describe(self) -> str:
        exceptions = self.total - self.matching
        lines = [
            f"automatic beaming matches the engraving   {self.matching:>9,}   {self.rate:6.1%}",
            f"exceptions the rule does not predict      {exceptions:>9,}   "
            f"{1 - self.rate:6.1%}",
        ]
        if self.disagreements:
            lines.append("")
            lines.append("largest disagreements (rule -> engraving):")
            for (rule, engraved), count in self.disagreements.most_common(6):
                lines.append(f"  {rule:>14} -> {engraved:<14} {count:>8,}")
        return "\n".join(lines)


def _flags(note: ET.Element) -> int:
    return applicable_beam_levels(note.findtext("type"))


def _duration(note: ET.Element) -> int:
    text = note.findtext("duration")
    return int(text) if text and text.strip().isdigit() else 0


def measure_part(part: ET.Element, baseline: Baseline) -> None:
    """Walk one part measure by measure, comparing the rule against the engraving.

    Onsets are accumulated per voice within each measure, because the rule beams within a
    voice: interleaving two voices' notes by document order would invent groups that
    cross between them.
    """
    extractor = NotationExtractor()
    divisions = DEFAULT_DIVISIONS
    beats, beat_type = DEFAULT_TIME

    for measure in part.findall("measure"):
        divisions_text = measure.findtext("attributes/divisions")
        if divisions_text and divisions_text.strip().isdigit():
            divisions = int(divisions_text)
        time = measure.find("attributes/time")
        if time is not None:
            beats_text = time.findtext("beats")
            type_text = time.findtext("beat-type")
            if beats_text and type_text and beats_text.isdigit() and type_text.isdigit():
                beats, beat_type = int(beats_text), int(type_text)

        voices: dict[str, list[tuple[BeamableNote, tuple[BeamLevelState, ...]]]] = {}
        onsets: dict[str, int] = {}
        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            engraved = extractor.extract(note)
            onset = onsets.get(voice, 0)
            duration = _duration(note)
            is_rest = note.find("rest") is not None
            # A chord's notes share an onset and one stem, so only the first carries the
            # beam decision; the rest would triple-count the same engraving choice.
            if note.find("chord") is None:
                voices.setdefault(voice, []).append(
                    (
                        BeamableNote(
                            onset=onset,
                            duration=duration,
                            flags=_flags(note),
                            is_rest=is_rest,
                        ),
                        engraved.beam_levels,
                    )
                )
                onsets[voice] = onset + duration

        beat = beat_divisions(beats, beat_type, divisions)
        wide = wide_unit(beats, beat_type, divisions)
        for entries in voices.values():
            notes = [note for note, _ in entries]
            predicted = automatic_beams(notes, beat, wide)
            for (note, engraved_vector), rule_vector in zip(entries, predicted, strict=True):
                if note.flags == 0 or note.is_rest:
                    continue
                _score(baseline, rule_vector, engraved_vector)

    extractor.close()


def _score(
    baseline: Baseline,
    rule: tuple[BeamLevelState, ...],
    engraved: tuple[BeamLevelState, ...],
) -> None:
    baseline.total += 1
    if tuple(rule) == tuple(engraved):
        baseline.matching += 1
        return
    for rule_state, engraved_state in zip(rule, engraved, strict=True):
        if rule_state != engraved_state:
            baseline.disagreements[(str(rule_state), str(engraved_state))] += 1


def measure(paths: list[Path]) -> Baseline:
    baseline = Baseline()
    for path in paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            print(f"skipping unparseable {path}")
            continue
        for part in root.findall("part"):
            measure_part(part, baseline)
    return baseline


def score_files(dataset_root: Path, track: str, split: str | None) -> list[Path]:
    """Whole-score MusicXML for one split, in a stable order.

    Whole scores, not the systemwise segments under `musicxml/unaligned/`. A segment cuts
    the page at a system break, which cuts beam groups and restarts the divisions and
    time-signature context, so a rule measured on segments is measured against fragments
    of its own input - it scored 91.9% on a sample of them against 79.4% on whole scores,
    which is the fragmentation flattering the rule, not the rule doing better.

    The `_cleaned` export is preferred where it exists: it is MuseScore's normalised
    round-trip, and therefore the notation that actually reaches the training labels.
    """
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
    parser.add_argument(
        "--split",
        help="Restrict to one split of the frozen manifest; omit for the whole corpus.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after N scores.")
    args = parser.parse_args()

    paths = score_files(args.dataset_root, args.track, args.split)
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} score file(s), split={args.split or 'all'}, track={args.track}")

    baseline = measure(paths)
    print(baseline.describe())


if __name__ == "__main__":
    main()
