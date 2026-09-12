"""Classify every place the engraving's beaming differs from the deterministic rule.

`beam_baseline` says *how often* the rule is wrong. This says *how* - which matters
because the fix depends on it. If the engraver's exceptions are mostly beams that span a
rest, the rule can be extended to cover them syntactically, and the head only has to be
trusted for that one identifiable class. If they are spread across many causes, no narrow
override reaches them and the head has to be trusted broadly - which the scanned crosstab
says it cannot be (it loses 12,027 notes the rule had right to recover 4,231).

Five buckets, and the split between the first two is the whole point:

  `beams_over_rest`      the engraving beams across a rest; the rule cannot express this
                         at all, because a rest ends a group by construction, so these
                         are a capability gap rather than a judgement the rule got wrong.
  `crosses_beat`         the engraving beams across a beat boundary the rule splits at.
  `other_engraved_beam`  the engraving beams where the rule flags, for neither reason.
  `unbeamed_exception`   the rule beams and the engraving flags - the case that cannot be
                         expressed by omission, since silence reads as "beam this".
  `different_grouping`   both beam, at different levels or states.
"""

# flake8: noqa: T201

import argparse
import json
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
from homr.transformer.structured_notation import BeamLevelState
from training.omr_datasets.beam_baseline import (
    DEFAULT_DIVISIONS,
    DEFAULT_TIME,
    _duration,
    _flags,
)
from training.omr_datasets.structured_notation_parser import NotationExtractor
from training.omr_datasets.unbeamed_review_set import JOINED, is_beamed, read_score


@dataclass
class Inventory:
    considered: int = 0
    agreeing: int = 0
    buckets: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[dict]] = field(default_factory=dict)

    def record(self, bucket: str, example: dict, keep: int = 40) -> None:
        self.buckets[bucket] += 1
        held = self.examples.setdefault(bucket, [])
        if len(held) < keep:
            held.append(example)


def _rest_inside_engraved_run(entries: list, index: int) -> bool:
    """Whether this note's engraved beam run has a rest inside it.

    Walks outward from `index` while the engraving keeps the level joined. A rest that is
    *inside* such a run is the case the rule cannot produce; a rest bounding the run is
    ordinary and is not counted.
    """
    left = index
    while left > 0:
        note, vector, _ = entries[left - 1]
        if note.is_rest:
            # keep walking: a rest inside a run is exactly what we are looking for
            if any(is_beamed(v) for n, v, _ in entries[: left - 1][::-1][:1]) or is_beamed(vector):
                left -= 1
                continue
            break
        if not is_beamed(vector):
            break
        left -= 1
    right = index
    while right < len(entries) - 1:
        note, vector, _ = entries[right + 1]
        if note.is_rest:
            if is_beamed(vector) or (right + 2 < len(entries) and is_beamed(entries[right + 2][1])):
                right += 1
                continue
            break
        if not is_beamed(vector):
            break
        right += 1
    return any(entries[i][0].is_rest for i in range(left, right + 1))


def _crosses_beat(entries: list, index: int, beat: int) -> bool:
    """Whether the engraved run this note belongs to spans a beat boundary."""
    if beat <= 0:
        return False
    left = index
    while left > 0 and is_beamed(entries[left - 1][1]) and not entries[left - 1][0].is_rest:
        left -= 1
    right = index
    while (
        right < len(entries) - 1
        and is_beamed(entries[right + 1][1])
        and not entries[right + 1][0].is_rest
    ):
        right += 1
    start = entries[left][0].onset
    end = entries[right][0].onset + entries[right][0].duration
    return (start // beat) != ((end - 1) // beat)


HOOKS = {BeamLevelState.FORWARD_HOOK, BeamLevelState.BACKWARD_HOOK}


def _run_extent(vectors: list, entries: list, index: int, level: int = 0) -> tuple[int, int]:
    """The span of the beam run containing `index`, judged at one level.

    A rest bounds the run here even though a run can contain one: this is used to compare
    two runs' *extents*, and treating a rest as a boundary on both sides compares like
    with like. `beams_over_rest` is classified before this is reached.
    """

    def state(i: int) -> BeamLevelState | None:
        vector = vectors[i]
        return vector[level] if level < len(vector) else None

    def joined(i: int) -> bool:
        return not entries[i][0].is_rest and state(i) in JOINED

    # `end` followed by `begin` is two runs, not one. Both states are "joined", so a
    # walker that only asks whether the neighbour is joined runs straight through the
    # boundary and reports one span where the engraving has two - which reads as two
    # runs of equal length in different places rather than as the subdivision it is.
    left = index
    while left > 0 and state(left) != BeamLevelState.BEGIN:
        if not joined(left - 1) or state(left - 1) == BeamLevelState.END:
            break
        left -= 1
    right = index
    while right < len(entries) - 1 and state(right) != BeamLevelState.END:
        if not joined(right + 1) or state(right + 1) == BeamLevelState.BEGIN:
            break
        right += 1
    return left, right


def _grouping_kind(
    rule_vector: tuple[BeamLevelState, ...], engraved_vector: tuple[BeamLevelState, ...]
) -> str:
    """Why two vectors that both beam still differ.

    The split that matters is `hook_vs_flag` against the rest. A hook is the partial beam
    on the short note of a dotted pair, and which of the two it is follows from the
    durations - so a rule that emits a flag there is simply not implementing the rule,
    not losing to information only the page carries. Separating it says how much of the
    apparent gap is arithmetic the baseline never did.
    """
    pairs = [
        (r, e)
        for r, e in zip(rule_vector, engraved_vector, strict=True)
        if not (r == BeamLevelState.NOT_APPLICABLE and e == BeamLevelState.NOT_APPLICABLE)
    ]
    differing = [(r, e) for r, e in pairs if r != e]
    if differing and all(
        (r == BeamLevelState.FLAG and e in HOOKS) or (e == BeamLevelState.FLAG and r in HOOKS)
        for r, e in differing
    ):
        return "hook_vs_flag"
    if all(
        (r in JOINED or r == BeamLevelState.NOT_APPLICABLE)
        and (e in JOINED or e == BeamLevelState.NOT_APPLICABLE)
        for r, e in pairs
    ):
        return "run_boundary"
    return "other_grouping"


def _boundary_kind(
    entries: list,
    predicted: list,
    index: int,
    rule_vector: tuple[BeamLevelState, ...],
    engraved_vector: tuple[BeamLevelState, ...],
) -> str:
    """Split a run-boundary disagreement into the kinds that need different answers.

    `secondary_subdivision` is the one to separate first: the primary beam agrees and only
    the shorter levels differ, which is an engraver subdividing secondary beams inside a
    group both sides already agree on. That is a rule the baseline could implement, not
    information from the image.

    The rest is a genuine disagreement about where the primary group starts and stops, and
    the direction says which way: a longer engraved run means the engraver joined what the
    rule split, shorter means the opposite.
    """
    primary_differs = rule_vector and engraved_vector and rule_vector[0] != engraved_vector[0]
    if not primary_differs:
        return "secondary_subdivision"
    rule_start, rule_end = _run_extent([tuple(v) for v in predicted], entries, index)
    eng_start, eng_end = _run_extent([v for _, v, _ in entries], entries, index)
    rule_len = rule_end - rule_start
    eng_len = eng_end - eng_start
    if eng_len > rule_len:
        return "primary_longer_engraved"
    if eng_len < rule_len:
        return "primary_shorter_engraved"
    return "primary_shifted"


def scan_part(part: ET.Element, score: str, part_name: str, inv: Inventory) -> None:
    extractor = NotationExtractor()
    divisions = DEFAULT_DIVISIONS
    beats, beat_type = DEFAULT_TIME

    for measure in part.findall("measure"):
        divisions_text = measure.findtext("attributes/divisions")
        if divisions_text and divisions_text.strip().isdigit():
            divisions = int(divisions_text)
        time = measure.find("attributes/time")
        if time is not None:
            b, t = time.findtext("beats"), time.findtext("beat-type")
            if b and t and b.isdigit() and t.isdigit():
                beats, beat_type = int(b), int(t)

        voices: dict[str, list] = {}
        onsets: dict[str, int] = {}
        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            engraved = extractor.extract(note)
            onset = onsets.get(voice, 0)
            if note.find("chord") is None:
                voices.setdefault(voice, []).append(
                    (
                        BeamableNote(
                            onset=onset,
                            duration=_duration(note),
                            flags=_flags(note),
                            is_rest=note.find("rest") is not None,
                        ),
                        engraved.beam_levels,
                        voice,
                    )
                )
                onsets[voice] = onset + _duration(note)

        beat = beat_divisions(beats, beat_type, divisions)
        wide = wide_unit(beats, beat_type, divisions)
        for voice, entries in voices.items():
            notes = [n for n, _, _ in entries]
            predicted = automatic_beams(notes, beat, wide)
            for index, ((note, engraved_vector, _), rule_vector) in enumerate(
                zip(entries, predicted, strict=True)
            ):
                if note.flags == 0 or note.is_rest:
                    continue
                inv.considered += 1
                if tuple(rule_vector) == tuple(engraved_vector):
                    inv.agreeing += 1
                    continue
                example = {
                    "score": score,
                    "part": part_name,
                    "measure": measure.get("number") or "?",
                    "voice": voice,
                    "onset": note.onset,
                    "rule": [str(s) for s in rule_vector],
                    "engraved": [str(s) for s in engraved_vector],
                }
                rule_b, eng_b = is_beamed(rule_vector), is_beamed(engraved_vector)
                if eng_b and not rule_b:
                    if _rest_inside_engraved_run(entries, index):
                        inv.record("beams_over_rest", example)
                    elif _crosses_beat(entries, index, beat):
                        inv.record("crosses_beat", example)
                    else:
                        inv.record("other_engraved_beam", example)
                elif rule_b and not eng_b:
                    inv.record("unbeamed_exception", example)
                else:
                    kind = _grouping_kind(rule_vector, engraved_vector)
                    if kind == "run_boundary":
                        kind = _boundary_kind(
                            entries, predicted, index, rule_vector, engraved_vector
                        )
                    inv.record(kind, example)
    extractor.close()


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
        except Exception:  # noqa: BLE001
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
                "buckets": dict(inv.buckets.most_common()),
                "examples": inv.examples,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"\n{inv.considered:,} flagged notes, {inv.agreeing:,} agree "
        f"({inv.agreeing / max(inv.considered, 1):.1%}), {disagreeing:,} differ\n"
    )
    print(f"{'bucket':<24} {'count':>9}  {'of differences':>14}")
    for name, count in inv.buckets.most_common():
        print(f"{name:<24} {count:>9,}  {count / max(disagreeing, 1):>13.1%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
