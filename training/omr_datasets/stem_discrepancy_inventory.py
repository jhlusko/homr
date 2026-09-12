"""Classify every place the engraved stem direction differs from the rule.

The same exercise `beam_discrepancy_inventory` does for beams, for the head with the most
existing evidence and the least explanation of it: `stem_arbiter` measures the head at
94.3% and the rule at 94.4%, failing on almost disjoint notes, with an oracle that would
reach 98.5%. Four points sit between "pick either" and "pick the right one each time", and
nothing currently says what those four points are made of.

Stem direction is the most rule-governed of the notations, which is exactly why the
residue is interesting: whatever the rule cannot reach is either a convention it has not
been taught or something only the image carries, and those want opposite responses.

Buckets, in the order they are tested - the order matters, because a note can satisfy
several and the earliest is the most specific explanation:

  `multi_voice`     the measure carries more than one voice, so direction is assigned by
                    voice rather than by pitch and the pitch rule is simply the wrong
                    rule here.
  `beam_group`      the note is inside a beamed group. An engraver sets one direction for
                    the whole group from its most extreme notehead, so an individual note
                    can legitimately point against its own position.
  `chord`           chord members share one stem, so only the extreme notehead chooses.
  `near_middle`     the notehead sits within a step of the middle line, where the choice
                    is conventionally free and engravers differ.
  `plain`           none of the above: a single-voice, unbeamed, unambiguous note whose
                    stem points the other way. This is the bucket that would justify a
                    head, and the one to look at first.
"""

# flake8: noqa: T201

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from homr.transformer.structured_notation import StemDirection
from training.omr_datasets.stem_baseline import (
    _CLEF_LINE,
    _predict,
    middle_line,
    note_position,
    stated_stem,
)
from training.omr_datasets.unbeamed_review_set import read_score

#: Within this many diatonic steps of the middle line, the direction is conventionally
#: the engraver's choice rather than the rule's.
NEAR_MIDDLE = 1


@dataclass
class Inventory:
    considered: int = 0
    agreeing: int = 0
    #: Notes the pitch rule alone gets wrong and the beam-group convention gets right.
    recovered_by_grouping: int = 0
    buckets: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict]] = field(default_factory=dict)

    def record(self, bucket: str, example: dict, keep: int = 40) -> None:
        self.buckets[bucket] += 1
        held = self.examples.setdefault(bucket, [])
        if len(held) < keep:
            held.append(example)


def _close_group(run: list[tuple[int, int]], group_direction: dict[int, StemDirection]) -> None:
    """Assign one direction to every note of a finished beam group.

    The engraver takes it from the notehead furthest from the middle line, so a note near
    the middle can legitimately point against its own position.
    """
    if not run:
        return
    extreme = max(run, key=lambda entry: abs(entry[1]))[1]
    for member, _ in run:
        group_direction[member] = _predict(extreme)


def scan_part(part: ET.Element, score: str, part_name: str, inv: Inventory) -> None:
    middle = middle_line("G", _CLEF_LINE["G"])

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

        voices = {note.findtext("voice") or "1" for note in measure.findall("note")}
        multi_voice = len(voices) > 1

        # The engraver sets one direction per beamed group, from its most extreme
        # notehead. Resolving that first separates "the rule was never taught this
        # convention" from "the rule cannot know"; `stem_baseline` already implements it
        # as its `grouped` variant, so measuring against the pitch rule alone credits a
        # head with recovering arithmetic the baseline could have done itself.
        group_direction: dict[int, StemDirection] = {}
        run: list[tuple[int, int]] = []

        for order, note in enumerate(measure.findall("note")):
            position = note_position(note, middle)
            beams = [(b.text or "").strip() for b in note.findall("beam")]
            if position is not None and beams:
                run.append((order, position))
                if "end" in beams:
                    _close_group(run, group_direction)
                    run = []
            else:
                _close_group(run, group_direction)
                run = []
        _close_group(run, group_direction)

        for order, note in enumerate(measure.findall("note")):
            actual = stated_stem(note)
            position = note_position(note, middle)
            if actual is None or position is None:
                continue
            inv.considered += 1
            pitch_only = _predict(position)
            predicted = group_direction.get(order, pitch_only)
            if pitch_only != actual and predicted == actual:
                inv.recovered_by_grouping += 1
            if predicted == actual:
                inv.agreeing += 1
                continue

            beams = [(b.text or "").strip() for b in note.findall("beam")]
            example = {
                "score": score,
                "part": part_name,
                "measure": measure.get("number") or "?",
                "voice": note.findtext("voice") or "1",
                "position": position,
                "predicted": str(predicted),
                "engraved": str(actual),
            }
            if multi_voice:
                inv.record("multi_voice", example)
            elif beams:
                inv.record("beam_group", example)
            elif note.find("chord") is not None:
                inv.record("chord", example)
            elif abs(position) <= NEAR_MIDDLE:
                inv.record("near_middle", example)
            else:
                inv.record("plain", example)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(p for p in args.scores.rglob("*") if p.suffix in {".mxl", ".musicxml", ".xml"})
    if args.limit:
        paths = paths[: args.limit]
    inv = Inventory()
    unreadable = 0
    for index, path in enumerate(paths, start=1):
        try:
            root = read_score(path)
        except Exception:  # noqa: BLE001, S112
            unreadable += 1
            continue
        if root is None:
            unreadable += 1
            continue
        for part in root.findall("part"):
            scan_part(part, path.stem, part.get("id") or "?", inv)
        if index % 20 == 0 or index == len(paths):
            print(f"  [{index}/{len(paths)}] {inv.considered:,} notes", flush=True)

    disagreeing = inv.considered - inv.agreeing
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "considered": inv.considered,
                "agreeing": inv.agreeing,
                "disagreeing": disagreeing,
                "recovered_by_grouping": inv.recovered_by_grouping,
                "buckets": dict(inv.buckets.most_common()),
                "examples": inv.examples,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"\n{inv.recovered_by_grouping:,} of the pitch rule's misses are recovered by the "
        f"beam-group convention alone"
    )
    print(
        f"\n{inv.considered:,} stemmed notes, {inv.agreeing:,} agree "
        f"({inv.agreeing / max(inv.considered, 1):.1%}), {disagreeing:,} differ\n"
    )
    print(f"{'bucket':<16} {'count':>9}  {'of differences':>14}  {'of all notes':>13}")
    for name, count in inv.buckets.most_common():
        print(
            f"{name:<16} {count:>9,}  {count / max(disagreeing, 1):>13.1%}"
            f"  {count / max(inv.considered, 1):>12.2%}"
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
