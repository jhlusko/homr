"""
Does beam materialization need to run on this corpus, and is it safe?

The design assumes absence of <beam> in MusicXML is ambiguous - automatic beaming or a
deliberate flag - and prescribes materializing the automatic choices into explicit beams
before using them as targets, with a check that materialization does not change the
rendered notation. This measures both halves of that on the actual corpus.

The materialization it tests is the round trip the design describes: hand the source to
the pinned MuseScore and take back what it writes. Two quantities matter.

  beams gained    notes that had no <beam> and came back with one. This is the ambiguity
                  the design is worried about. If it is large, absence meant "beam me
                  automatically" and the FLAG labels are wrong without materialization.

  beams changed   notes whose beam vector came back different. This is the risk running
                  materialization anyway: a round trip that rewrites grouping corrupts
                  labels it was supposed to make explicit.

Measured on 14 OSSQ scores, 172,607 notes: 1 gained, 775 lost, 2,135 vectors changed.
Nothing is ambiguous - one note in 172,607 - so materialization has nothing to fix here,
and running it would rewrite grouping on 1.7% of notes. The FLAG labels stand as read.

That conclusion is corpus- and version-specific, which is why this is a repeatable check
and records the MuseScore version rather than a one-off note in a commit message.
"""

# flake8: noqa: T201

import argparse
import collections
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Above this share of notes gaining beams, absence in the source meant "beam
#: automatically" and materialization is required before the labels can be trusted.
AMBIGUITY_THRESHOLD = 0.005

#: Above this share of notes whose vector changes, a MuseScore round trip is rewriting
#: grouping rather than preserving it, so it must not be used to produce labels.
INSTABILITY_THRESHOLD = 0.005

BeamVector = tuple[tuple[str, str], ...]


@dataclass
class RoundTripResult:
    notes: int = 0
    unchanged: int = 0
    gained: int = 0
    lost: int = 0
    changed: int = 0
    skipped_scores: list[str] = field(default_factory=list)
    changes: collections.Counter[tuple[str, BeamVector, BeamVector]] = field(
        default_factory=collections.Counter
    )

    def add(self, other: "RoundTripResult") -> None:
        self.notes += other.notes
        self.unchanged += other.unchanged
        self.gained += other.gained
        self.lost += other.lost
        self.changed += other.changed
        self.skipped_scores.extend(other.skipped_scores)
        self.changes.update(other.changes)

    @property
    def ambiguity_rate(self) -> float:
        return self.gained / self.notes if self.notes else 0.0

    @property
    def instability_rate(self) -> float:
        return (self.lost + self.changed) / self.notes if self.notes else 0.0

    @property
    def materialization_needed(self) -> bool:
        return self.ambiguity_rate > AMBIGUITY_THRESHOLD

    @property
    def round_trip_safe(self) -> bool:
        return self.instability_rate <= INSTABILITY_THRESHOLD

    def verdict(self) -> str:
        if self.materialization_needed and self.round_trip_safe:
            return "materialization IS needed and the round trip preserves grouping: run it"
        if self.materialization_needed:
            return (
                "materialization is needed but the round trip rewrites grouping: "
                "a different materialization source is required"
            )
        if not self.round_trip_safe:
            return (
                "no ambiguity to resolve, and the round trip rewrites grouping: "
                "skip materialization, and do not use a MuseScore round trip on labels"
            )
        return "no ambiguity to resolve: skip materialization"


def beam_vectors(path: Path) -> list[tuple[str | None, BeamVector]]:
    """Per sounding note, its (type, beam vector), in document order.

    Chord members carry no beams of their own and rests none at all, so both are skipped
    to keep the two sides of the comparison aligned by note.
    """
    root = ET.parse(path).getroot()  # noqa: S314
    notes: list[tuple[str | None, BeamVector]] = []
    for part in root.findall("part"):
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                if note.find("chord") is not None or note.find("rest") is not None:
                    continue
                vector = tuple(
                    sorted(
                        (beam.get("number") or "", (beam.text or "").strip())
                        for beam in note.findall("beam")
                    )
                )
                notes.append((note.findtext("type"), vector))
    return notes


def compare(
    before: Sequence[tuple[str | None, BeamVector]],
    after: Sequence[tuple[str | None, BeamVector]],
) -> RoundTripResult:
    """Classify every note's beam vector across a round trip.

    A note-count mismatch means the two sides cannot be compared note by note, so the
    score is reported rather than aligned by guesswork.
    """
    result = RoundTripResult()
    if len(before) != len(after):
        result.skipped_scores.append(f"note count changed {len(before)} -> {len(after)}")
        return result
    result.notes = len(before)
    for (note_type, first), (_, second) in zip(before, after, strict=True):
        if first == second:
            result.unchanged += 1
        elif not first and second:
            result.gained += 1
            result.changes[(note_type or "?", first, second)] += 1
        elif first and not second:
            result.lost += 1
            result.changes[(note_type or "?", first, second)] += 1
        else:
            result.changed += 1
            result.changes[(note_type or "?", first, second)] += 1
    return result


def musescore_version() -> str:
    try:
        done = subprocess.run(  # noqa: S603
            ["xvfb-run", "-a", "mscore", "--version"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable ({error})"
    return (
        (done.stdout or done.stderr).strip().splitlines()[-1]
        if done.stdout or done.stderr
        else "unknown"
    )


def round_trip(source: Path) -> RoundTripResult:
    """Export the score through MuseScore and compare beams with the original."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "round_trip.musicxml"
        done = subprocess.run(  # noqa: S603
            ["xvfb-run", "-a", "mscore", "-o", str(target), str(source)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        if done.returncode != 0 or not target.is_file():
            result = RoundTripResult()
            result.skipped_scores.append(f"{source.name}: MuseScore export failed")
            return result
        return compare(beam_vectors(source), beam_vectors(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("scores", type=Path, nargs="+", help="Original MusicXML files.")
    parser.add_argument("--show-changes", type=int, default=8, help="Top N change patterns.")
    args = parser.parse_args()

    print(f"MuseScore: {musescore_version()}")
    total = RoundTripResult()
    for source in args.scores:
        result = round_trip(source)
        total.add(result)
        if result.skipped_scores:
            print(f"{source.name}: {result.skipped_scores[-1]}")
        else:
            print(
                f"{source.name}: {result.unchanged} unchanged, {result.gained} gained, "
                f"{result.lost} lost, {result.changed} changed"
            )

    print()
    print(f"notes compared      {total.notes:,}")
    print(f"beams gained        {total.gained:,}  ({100 * total.ambiguity_rate:.3f}%)")
    print(
        f"beams lost/changed  {total.lost + total.changed:,}  ({100 * total.instability_rate:.3f}%)"
    )
    if args.show_changes:
        print("\nmost common changes (type, before -> after):")
        for (note_type, first, second), count in total.changes.most_common(args.show_changes):
            before = [state for _, state in first]
            after = [state for _, state in second]
            print(f"  {count:6d}  {note_type:8s} {before} -> {after}")
    print(f"\n=> {total.verdict()}")


if __name__ == "__main__":
    main()
