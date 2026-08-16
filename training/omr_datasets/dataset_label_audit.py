"""
Per-class support in the built training set, as the heads will actually see it.

`ossq_label_audit.py` counts the corpus. This counts the dataset, and the two differ in
ways that bear directly on the head configuration §25.2 leaves open:

  - the training unit is one system's staff, so slur slots are canonicalised per system
    rather than per score. Fewer spans are concurrently open in a system than in a whole
    movement, so slot usage here is lower than the corpus figure, and the corpus figure
    is the wrong one to size heads against.
  - systems whose staff crops do not line up with their parts are skipped whole
    (27.11), so some of the corpus is simply not in the training set.
  - a slur crossing a system break now contributes a STOP where it previously
    contributed nothing (27.17).

Counts come from the sidecars, which are the labels training reads, rather than from
re-parsing MusicXML - so this measures the artefact, not an intention about it.

A head is only worth building where the class it must learn actually occurs. The report
names classes with no support at all, because a head trained on none of a class will
still emit that logit and can never be right about it.
"""

# flake8: noqa: T201

import argparse
import collections
from dataclasses import dataclass, field
from pathlib import Path

from homr.transformer.structured_notation import (
    BeamLevelState,
    SlurEvent,
    SlurSide,
    StemDirection,
)
from training.omr_datasets.notation_sidecar import SidecarMismatch, attach_sidecar
from training.transformer.training_vocabulary import read_tokens


@dataclass
class DatasetCounts:
    examples: int = 0
    annotated: int = 0
    notes: int = 0
    #: level -> state -> count, over notes the level applies to
    beam_states: dict[int, collections.Counter[str]] = field(default_factory=dict)
    stems: collections.Counter[str] = field(default_factory=collections.Counter)
    slur_events: dict[int, collections.Counter[str]] = field(default_factory=dict)
    slur_sides: collections.Counter[str] = field(default_factory=collections.Counter)

    def observe(self, notation: object) -> None:
        self.notes += 1
        beams = notation.beam_levels  # type: ignore[attr-defined]
        for level, state in enumerate(beams, start=1):
            if state == BeamLevelState.NOT_APPLICABLE:
                continue
            self.beam_states.setdefault(level, collections.Counter())[str(state)] += 1
        self.stems[str(notation.stem)] += 1  # type: ignore[attr-defined]
        for slot, (event, side) in enumerate(notation.slurs, start=1):  # type: ignore[attr-defined]
            if event == SlurEvent.NONE:
                continue
            self.slur_events.setdefault(slot, collections.Counter())[str(event)] += 1
            if side != SlurSide.UNSPECIFIED:
                self.slur_sides[str(side)] += 1


def audit_index(index_path: Path) -> tuple[DatasetCounts, list[str]]:
    """Count every label behind one index file, and report sidecars that do not fit.

    Labels are read through `attach_sidecar` rather than straight out of the JSON, so
    every sidecar is checked against its own token file on the way past: if the writer
    and reader disagree about which symbols are note-bearing, one note's beams would land
    on another. A whole-dataset pass is the cheapest place to find that.

    Mismatches are collected rather than raised. An audit that dies on the first bad file
    tells you less than one that tells you how many there are.
    """
    counts = DatasetCounts()
    problems: list[str] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        counts.examples += 1
        _, _, token_path = line.partition(",")
        symbols = read_tokens(token_path.strip())
        try:
            attached = attach_sidecar(token_path.strip(), symbols)
        except SidecarMismatch as mismatch:
            problems.append(str(mismatch))
            continue
        if not attached:
            continue
        counts.annotated += 1
        for symbol in symbols:
            if symbol.notation is not None:
                counts.observe(symbol.notation)
    return counts, problems


def _levels_table(counts: DatasetCounts) -> str:
    lines = ["beam levels (over notes the level applies to):"]
    states = [s for s in BeamLevelState if s != BeamLevelState.NOT_APPLICABLE]
    header = "  level" + "".join(f"{str(s)[:9]:>11}" for s in states) + f"{'total':>11}"
    lines.append(header)
    for level in sorted(counts.beam_states):
        row = counts.beam_states[level]
        total = sum(row.values())
        lines.append(
            f"  {level:>5}" + "".join(f"{row[str(s)]:>11,}" for s in states) + f"{total:>11,}"
        )
    return "\n".join(lines)


def _slur_table(counts: DatasetCounts) -> str:
    lines = ["slur slots (over notes carrying an event):"]
    events = [e for e in SlurEvent if e != SlurEvent.NONE]
    lines.append("  slot" + "".join(f"{str(e)[:13]:>15}" for e in events))
    for slot in sorted(counts.slur_events):
        row = counts.slur_events[slot]
        lines.append(f"  {slot:>4}" + "".join(f"{row[str(e)]:>15,}" for e in events))
    return "\n".join(lines)


def describe(counts: DatasetCounts) -> str:
    stems = "  ".join(
        f"{name}={count:,}" for name, count in counts.stems.most_common() if count
    )
    sides = "  ".join(f"{name}={count:,}" for name, count in counts.slur_sides.most_common())
    parts = [
        f"{counts.examples:,} examples, {counts.annotated:,} with labels, "
        f"{counts.notes:,} annotated notes",
        "",
        _levels_table(counts),
        "",
        f"stems: {stems or 'none'}",
        "",
        _slur_table(counts),
        f"slur placement: {sides or 'none stated'}",
    ]
    return "\n".join(parts)


def unsupported(counts: DatasetCounts, beam_levels: int, slur_slots: int) -> list[str]:
    """Configured heads that this dataset cannot train.

    A head with no targets still emits logits, and the manifest would be free to declare
    it if the run never noticed. Naming them here means the configuration is a decision
    rather than an oversight.
    """
    missing = []
    for level in range(1, beam_levels + 1):
        if not counts.beam_states.get(level):
            missing.append(f"beam.level.{level}")
    for slot in range(1, slur_slots + 1):
        if not counts.slur_events.get(slot):
            missing.append(f"slur.slot.{slot}")
    if not any(
        count for name, count in counts.stems.items() if name != str(StemDirection.UNKNOWN)
    ):
        missing.append("stem.direction")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument("--beam-levels", type=int, default=4)
    parser.add_argument("--slur-slots", type=int, default=2)
    args = parser.parse_args()

    counts, problems = audit_index(args.index)
    print(describe(counts))

    missing = unsupported(counts, args.beam_levels, args.slur_slots)
    if missing:
        print()
        print("configured but unsupported by this dataset: " + ", ".join(missing))
    if problems:
        print()
        print(f"{len(problems)} sidecar(s) do not match their token file:")
        for problem in problems[:5]:
            print(f"  {problem}")


if __name__ == "__main__":
    main()
