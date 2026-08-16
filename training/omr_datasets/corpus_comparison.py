"""
Is OSSQ representative? Compare notation statistics across corpora.

Every number this work has produced - the 82% automatic-beaming baseline, the per-class
supports that sized the heads, Gate C's 78.7% - comes from 122 string quartets. That is
one instrumentation, one texture, and a narrow slice of engraving practice. If PDMX's
250,000 scores beam or slur differently, those numbers describe quartets rather than
notation, and the head configuration was sized against the wrong distribution.

This answers that from the symbolic files alone. No rendering, no images, no conversion:
just the statistics that determine whether a head is worth having and what it has to beat.

Comparable by construction - the same extractor and the same rule that produced the OSSQ
figures are used here, so a difference is a difference in the music rather than in the
measurement.
"""

# flake8: noqa: T201

import argparse
import random
import xml.etree.ElementTree as ET
import zipfile
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

DEFAULT_TIME = (4, 4)


@dataclass
class Profile:
    """What a corpus's notation looks like, in the terms the heads are sized by."""

    name: str
    scores: int = 0
    notes: int = 0
    beamable: int = 0
    beam_states: Counter[str] = field(default_factory=Counter)
    rule_matches: int = 0
    rule_total: int = 0
    ties: int = 0
    slurs: int = 0
    slurs_with_placement: int = 0
    stems_stated: int = 0
    hooks: int = 0

    def rate(self, value: int, base: int) -> str:
        return f"{value / base:6.1%}" if base else "   n/a"

    def describe(self) -> str:
        return "\n".join(
            [
                f"{self.name}: {self.scores:,} scores, {self.notes:,} notes",
                f"  beamable notes            {self.beamable:>10,} "
                f"{self.rate(self.beamable, self.notes)} of notes",
                f"  automatic beaming matches {self.rule_matches:>10,} "
                f"{self.rate(self.rule_matches, self.rule_total)} of beamable",
                f"  hooks                     {self.hooks:>10,} "
                f"{self.rate(self.hooks, self.beamable)} of beamable",
                f"  ties                      {self.ties:>10,} "
                f"{self.rate(self.ties, self.notes)} of notes",
                f"  slurs                     {self.slurs:>10,} "
                f"{self.rate(self.slurs, self.notes)} of notes",
                f"  slurs stating placement   {self.slurs_with_placement:>10,} "
                f"{self.rate(self.slurs_with_placement, self.slurs)} of slurs",
                f"  notes stating a stem      {self.stems_stated:>10,} "
                f"{self.rate(self.stems_stated, self.notes)} of notes",
            ]
        )


def read_score(path: Path) -> ET.Element | None:
    """Parse a .musicxml or a zipped .mxl."""
    try:
        if path.suffix == ".mxl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                inner = next(
                    (n for n in names if n.endswith(".xml") and "/" not in n),
                    next((n for n in names if n.endswith(".xml")), None),
                )
                if inner is None:
                    return None
                return ET.fromstring(archive.read(inner).decode("utf-8"))  # noqa: S314
        return ET.parse(path).getroot()  # noqa: S314
    except Exception:  # noqa: BLE001
        return None


def profile_part(part: ET.Element, profile: Profile) -> None:
    extractor = NotationExtractor()
    divisions, (beats, beat_type) = 1, DEFAULT_TIME

    for measure in part.findall("measure"):
        text = measure.findtext("attributes/divisions")
        if text and text.strip().isdigit():
            divisions = int(text)
        time = measure.find("attributes/time")
        if time is not None:
            beats_text, type_text = time.findtext("beats"), time.findtext("beat-type")
            if beats_text and type_text and beats_text.isdigit() and type_text.isdigit():
                beats, beat_type = int(beats_text), int(type_text)

        by_voice: dict[str, list] = {}
        onsets: dict[str, int] = {}
        for note in measure.findall("note"):
            profile.notes += 1
            notation = extractor.extract(note)
            voice = note.findtext("voice") or "1"
            is_rest = note.find("rest") is not None

            for notations in note.findall("notations"):
                profile.ties += len(notations.findall("tied"))
                for slur in notations.findall("slur"):
                    profile.slurs += 1
                    if slur.get("placement") or slur.get("orientation"):
                        profile.slurs_with_placement += 1
            if note.findtext("stem"):
                profile.stems_stated += 1

            flags = applicable_beam_levels(note.findtext("type"))
            if flags and not is_rest:
                profile.beamable += 1
                for state in notation.beam_levels:
                    if state != BeamLevelState.NOT_APPLICABLE:
                        profile.beam_states[str(state)] += 1
                        if "hook" in str(state):
                            profile.hooks += 1

            duration_text = note.findtext("duration")
            duration = int(duration_text) if duration_text and duration_text.isdigit() else 0
            if note.find("chord") is None:
                onset = onsets.get(voice, 0)
                by_voice.setdefault(voice, []).append(
                    (
                        BeamableNote(onset, duration, flags, is_rest),
                        notation.beam_levels,
                    )
                )
                onsets[voice] = onset + duration

        beat = beat_divisions(beats, beat_type, divisions)
        wide = wide_unit(beats, beat_type, divisions)
        for entries in by_voice.values():
            predicted = automatic_beams([n for n, _ in entries], beat, wide)
            for (note, engraved), rule in zip(entries, predicted, strict=True):
                if note.flags == 0 or note.is_rest:
                    continue
                profile.rule_total += 1
                profile.rule_matches += tuple(rule) == tuple(engraved)
    extractor.close()


def profile_corpus(name: str, paths: list[Path]) -> Profile:
    profile = Profile(name)
    for path in paths:
        root = read_score(path)
        if root is None:
            continue
        profile.scores += 1
        for part in root.findall("part"):
            profile_part(part, profile)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--ossq-root", type=Path, required=True)
    parser.add_argument("--pdmx-root", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200, help="Scores per corpus.")
    args = parser.parse_args()

    ossq = [
        path
        for directory in sorted(args.ossq_root.glob("scores/*/*"))
        for path in sorted(directory.glob("*.musicxml"))
        if not path.stem.endswith("_cleaned")
    ]
    pdmx = sorted(args.pdmx_root.glob("**/*.mxl"))
    random.Random(0).shuffle(pdmx)
    print(f"OSSQ: {len(ossq):,} scores available; PDMX: {len(pdmx):,}")

    for profile in (
        profile_corpus("OSSQ", ossq[: args.sample]),
        profile_corpus("PDMX", pdmx[: args.sample]),
    ):
        print()
        print(profile.describe())


if __name__ == "__main__":
    main()
