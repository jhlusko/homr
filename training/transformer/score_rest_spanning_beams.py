"""Score the head's precision on rest-spanning beams - the A3 beam gate.

The gate (`OTS_HOMR_PUBLIC_RELEASE_ROADMAP.md` §1): the beam rule structurally cannot
beam across a rest, because a rest ends its group, so the proposed hybrid is "the rule
everywhere, the head only where it beams across a rest". That is only safe if the head is
*right* when it does, so the number is precision on the head's rest-spanning groups.

Input is `dump_rest_predictions`' JSONL: per staff, the symbol kinds in token order and
the level-1 predicted and reference beam states per position. A group is a run
`begin`, `continue`*, `end` over the staff's notes in token order; it spans a rest when a
rest symbol sits strictly between its first and last note. Chord members after the first
(a note right after a `chord` marker) and grace notes are skipped: they sound with, or
decorate, a note already in the run, and counting them turned every chord inside a group
into a second `begin` (2026-09-25: 29,177 of PDMX's malformed *reference* runs, 5,776
after skipping them). Anything else inside an open run (a flag, a hook, an unbeamed note,
a second `begin`) breaks it, and the partial run is counted as malformed rather than
repaired. What remains malformed is mostly two voices interleaved as chord pairs, which
token order cannot separate. So precision is also reported on **clean staves**, whose
reference parses with no malformed run.

Two references, reported side by side, because each can be wrong in its own way:

- **MusicXML reference.** Exact match of the group's first and last positions against the
  reference groups built from the same sidecar the heads were trained on. This is what
  "the head reproduced the score's beaming" means, but it inherits whatever the exporter
  did with beams across rests.
- **MuseScore source.** Every rest the predicted group spans carries an inside-the-beam
  `<BeamMode>` (`INSIDE_BEAM_MODES`) in the `.mscx` (`beam_placement.BeamPlacementIndex`,
  validated against real scores). This is independent of the MusicXML exporter. It is
  only available for OSSQ crops whose part passes that index's alignment gates, and only
  when the crop's rest count equals the joined slice's; other crops are counted as
  unjoined and left out.

Recall against each reference and the agreement between the two references are reported
too, so a precision figure never stands without its support.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REST = "r"
NOTE = "n"
CHORD = "c"

#: MuseScore rest `BeamMode`s that put the rest *between* beamed notes. `begin32` and
#: `begin64` break only a secondary beam, so the primary run still passes through the
#: rest. `begin`/`end` start or finish a beam *at* the rest - a rest at the group's edge,
#: never strictly inside it, so they cannot support a group that spans the rest.
INSIDE_BEAM_MODES = frozenset({"mid", "begin32", "begin64"})


def beam_groups(kinds: str, states: list[str]) -> tuple[list[tuple[int, int]], int]:
    """Level-1 groups as `(first, last)` symbol positions, plus the malformed-run count."""
    groups: list[tuple[int, int]] = []
    malformed = 0
    start: int | None = None
    for position, kind in enumerate(kinds):
        if kind != NOTE or (position > 0 and kinds[position - 1] == CHORD):
            continue
        state = states[position]
        if state == "begin":
            if start is not None:
                malformed += 1
            start = position
        elif state == "continue":
            if start is None:
                malformed += 1
        elif state == "end":
            if start is None:
                malformed += 1
            else:
                groups.append((start, position))
                start = None
        elif start is not None:
            malformed += 1
            start = None
    if start is not None:
        malformed += 1
    return groups, malformed


def spanned_rests(kinds: str, group: tuple[int, int]) -> list[int]:
    """Rest ordinals (the k-th rest of the staff) strictly inside the group."""
    ordinals = []
    rest_ordinal = -1
    for position, kind in enumerate(kinds):
        if kind != REST:
            continue
        rest_ordinal += 1
        if group[0] < position < group[1]:
            ordinals.append(rest_ordinal)
    return ordinals


def crop_key(token_path: str) -> tuple[str, int, int, int] | None:
    """`score_page_system_part` (part 1-based, as `convert_ossq` names crops)."""
    fields = Path(token_path).stem.split("_")
    if len(fields) < 4:
        return None
    try:
        page, system, part = int(fields[-3]), int(fields[-2]), int(fields[-1])
    except ValueError:
        return None
    return "_".join(fields[:-3]), page, system, part - 1


class MscxTruth:
    """Per-crop rest `BeamMode`, built lazily per score from `BeamPlacementIndex`."""

    def __init__(self, work: Path, mscx_dir: Path) -> None:
        self.work = work
        self.mscx_dir = mscx_dir
        self.indexes: dict[str, Any] = {}

    def rests(self, token_path: str) -> list[str | None] | None:
        key = crop_key(token_path)
        if key is None:
            return None
        score, page, system, part_index = key
        if score not in self.indexes:
            from training.omr_datasets.beam_placement import BeamPlacementIndex

            whole = self.work / f"{score}.musicxml"
            mscx = self.mscx_dir / f"{score}.mscx"
            self.indexes[score] = (
                BeamPlacementIndex(self.work, score, whole, mscx)
                if whole.is_file() and mscx.is_file()
                else None
            )
        index = self.indexes[score]
        if index is None:
            return None
        found = index.for_segment(page, system, part_index)
        return None if found is None else [mode for _is_rest, mode in found]


def current_kinds(record: dict) -> str:
    """The record's kinds, re-derived from its token file when that is readable.

    Records written before chord and grace kinds existed say `n`/`o` for them; the token
    file is the authority either way, and re-reading it keeps old runs scoreable without
    a GPU re-run.
    """
    if Path(record["tokens"]).is_file():
        from training.transformer.dump_rest_predictions import symbol_kinds

        return "".join(symbol_kinds(record["tokens"], len(record["kinds"])))
    return record["kinds"]


def score(records: list[dict], truth: MscxTruth | None) -> dict:
    counts: Counter[str] = Counter()
    for record in records:
        kinds = current_kinds(record)
        predicted, bad_predicted = beam_groups(kinds, record["predicted_beam"][0])
        reference, bad_reference = beam_groups(kinds, record["reference_beam"][0])
        counts["staves"] += 1
        counts["malformed_predicted_runs"] += bad_predicted
        counts["malformed_reference_runs"] += bad_reference
        predicted_spanning = [g for g in predicted if spanned_rests(kinds, g)]
        reference_spanning = [g for g in reference if spanned_rests(kinds, g)]
        reference_set = set(reference)
        counts["predicted_spanning"] += len(predicted_spanning)
        counts["reference_spanning"] += len(reference_spanning)
        counts["predicted_spanning_exact"] += sum(g in reference_set for g in predicted_spanning)
        if bad_reference == 0:
            counts["clean_staves"] += 1
            counts["clean_predicted_spanning"] += len(predicted_spanning)
            counts["clean_predicted_spanning_exact"] += sum(
                g in reference_set for g in predicted_spanning
            )
        counts["reference_spanning_found"] += sum(
            g in set(predicted) for g in reference_spanning
        )

        modes = truth.rests(record["tokens"]) if truth is not None else None
        rest_total = kinds.count(REST)
        if modes is None or len(modes) != rest_total:
            counts["staves_unjoined"] += 1
            continue
        counts["staves_joined"] += 1
        counts["joined_predicted_spanning"] += len(predicted_spanning)
        counts["joined_predicted_spanning_supported"] += sum(
            all(modes[k] in INSIDE_BEAM_MODES for k in spanned_rests(kinds, g))
            for g in predicted_spanning
        )
        # MuseScore's "inside a beam" rests, and whether the MusicXML reference agrees.
        inside = [k for k, mode in enumerate(modes) if mode in INSIDE_BEAM_MODES]
        covered_by_reference = {k for g in reference_spanning for k in spanned_rests(kinds, g)}
        covered_by_prediction = {k for g in predicted_spanning for k in spanned_rests(kinds, g)}
        counts["mscx_beamed_rests"] += len(inside)
        counts["mscx_beamed_rests_in_reference_group"] += sum(k in covered_by_reference for k in inside)
        counts["mscx_beamed_rests_in_predicted_group"] += sum(k in covered_by_prediction for k in inside)

    def ratio(numerator: str, denominator: str) -> float | None:
        return counts[numerator] / counts[denominator] if counts[denominator] else None

    return {
        "counts": dict(counts),
        "precision_vs_musicxml_reference": ratio("predicted_spanning_exact", "predicted_spanning"),
        "recall_vs_musicxml_reference": ratio("reference_spanning_found", "reference_spanning"),
        "precision_vs_musicxml_reference_clean_staves": ratio(
            "clean_predicted_spanning_exact", "clean_predicted_spanning"
        ),
        "precision_vs_mscx_beammode": ratio(
            "joined_predicted_spanning_supported", "joined_predicted_spanning"
        ),
        "recall_vs_mscx_beammode": ratio(
            "mscx_beamed_rests_in_predicted_group", "mscx_beamed_rests"
        ),
        "reference_agreement_with_mscx": ratio(
            "mscx_beamed_rests_in_reference_group", "mscx_beamed_rests"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rest-predictions", type=Path, required=True)
    parser.add_argument(
        "--work", type=Path, help="OSSQ build work dir: {score}.musicxml and musicxml/scanned/systemwise"
    )
    parser.add_argument("--mscx-dir", type=Path, help="Directory of {score}.mscx; default --work")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.rest_predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    truth = MscxTruth(args.work, args.mscx_dir or args.work) if args.work else None
    report = score(records, truth)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
