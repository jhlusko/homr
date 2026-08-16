"""
Can anything choose between the stem head and the beam-derived rule?

27.27 measured them at 94.3% and 94.4%, which looks like redundancy until the crosstab:
they fail on almost disjoint notes. The head rescues 72.7% of the rule's mistakes, the
rule rescues about as many of the head's, and only 1,690 notes of 111,229 defeat both. An
oracle choosing the better source each time would score 98.5%.

That 4-point gap is only reachable if something can tell which source to trust without
knowing the answer. Two signals are available, and this measures both:

  head confidence   the stem head's own softmax. Use the head when it is sure, the rule
                    otherwise.
  distance          how far the beam group's extreme notehead sits from the middle line.
                    The rule is a threshold on that distance, so it is least reliable
                    where the distance is smallest.

**The threshold is tuned on one half of the staves and reported on the other.** Choosing a
threshold on the same notes it is quoted against would report the tuning, not the policy -
and `test_synth` is reserved, so the split has to come out of validation.
"""

# flake8: noqa: T201

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from homr.transformer.structured_notation import StemDirection
from training.omr_datasets.stem_baseline import _predict
from training.transformer.derived_stems import (
    _vectors,
    derive,
    groups_from_beams,
    walk_part,
)
from training.transformer.rule_vs_head import _ordering, segment_for


@dataclass
class Row:
    """One note, with everything a policy may look at and the answer it is judged against."""

    actual: str
    head: str
    confidence: float
    rule: str
    #: Distance of the note's beam group from the middle line, the rule's own margin.
    margin: int


@dataclass
class Policy:
    name: str
    correct: int = 0
    total: int = 0
    took_head: int = 0

    @property
    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def describe(self) -> str:
        share = self.took_head / self.total if self.total else 0.0
        return f"  {self.name:<34} {self.rate:6.2%}   head used on {share:5.1%} of notes"


@dataclass
class Sweep:
    rows: list[Row] = field(default_factory=list)

    def evaluate(self, chooser, name: str) -> Policy:  # noqa: ANN001
        policy = Policy(name)
        for row in self.rows:
            use_head = chooser(row)
            policy.total += 1
            policy.took_head += use_head
            policy.correct += (row.head if use_head else row.rule) == row.actual
        return policy

    def oracle(self) -> Policy:
        return self.evaluate(lambda row: row.head == row.actual, "oracle (upper bound)")


def collect(predictions: Path, dataset_root: Path, levels: int) -> list[list[Row]]:
    """One list of rows per staff, so the tuning split can be made by staff.

    Splitting by note would put the same staff on both sides, and staves are the unit a
    threshold could overfit to.
    """
    staves: list[list[Row]] = []
    records = [
        json.loads(line)
        for line in predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in sorted(records, key=_ordering):
        located = segment_for(Path(record["tokens"]), dataset_root)
        if located is None:
            continue
        segment_path, part_index = located
        try:
            parts = ET.parse(segment_path).getroot().findall("part")  # noqa: S314
        except ET.ParseError:
            continue
        if part_index >= len(parts):
            continue

        notes, beamable = walk_part(parts[part_index], levels)
        head = record.get("stem_predicted") or []
        actual = record.get("stem_reference") or []
        confidence = record.get("stem_confidence") or []
        if not (len(head) == len(actual) == len(notes)) or len(beamable) != len(
            record["reference"]
        ):
            continue

        vectors = _vectors(record, "predicted", notes, levels)
        rule = derive(notes, vectors)
        margins = _margins(notes, vectors)

        staves.append(
            [
                Row(
                    actual=actual[index],
                    head=head[index],
                    confidence=confidence[index] if index < len(confidence) else 0.0,
                    rule=str(rule[index]),
                    margin=margins[index],
                )
                for index in range(len(notes))
            ]
        )
    return staves


def _margins(notes: list, vectors: list) -> list[int]:
    """Each note's distance from the middle line, taken from its beam group's extreme.

    That extreme is exactly what the rule thresholds on, so its size is how much room the
    rule had - a group whose furthest notehead sits on the line was a coin flip.
    """
    margins = [0] * len(notes)
    for group in groups_from_beams(vectors):
        extreme = max((notes[i].position for i in group), key=abs)
        for index in group:
            margins[index] = abs(extreme)
    return margins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    staves = collect(args.predictions, args.dataset_root, args.levels)
    half = len(staves) // 2
    tune = Sweep([row for stave in staves[:half] for row in stave])
    report = Sweep([row for stave in staves[half:] for row in stave])
    print(f"{len(staves):,} staves: {len(tune.rows):,} notes to tune on, "
          f"{len(report.rows):,} to report on")

    fixed = {
        "head alone": lambda row: True,
        "rule alone": lambda row: False,
    }
    print("\non the reporting half:")
    for name, chooser in fixed.items():
        print(report.evaluate(chooser, name).describe())
    print(report.oracle().describe())

    print("\ntuning:")
    best_confidence = max(
        (tune.evaluate(lambda row, t=t: row.confidence >= t, f"head if confidence >= {t}") for t in
         (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)),
        key=lambda policy: policy.rate,
    )
    best_margin = max(
        (tune.evaluate(lambda row, k=k: row.margin <= k, f"head if margin <= {k}") for k in
         range(0, 6)),
        key=lambda policy: policy.rate,
    )
    print(f"  best by confidence: {best_confidence.name} -> {best_confidence.rate:.2%}")
    print(f"  best by margin:     {best_margin.name} -> {best_margin.rate:.2%}")

    threshold = float(best_confidence.name.rsplit(" ", 1)[1])
    limit = int(best_margin.name.rsplit(" ", 1)[1])
    print("\nthose thresholds, applied to the reporting half:")
    print(report.evaluate(lambda row: row.confidence >= threshold, best_confidence.name).describe())
    print(report.evaluate(lambda row: row.margin <= limit, best_margin.name).describe())
    print(
        report.evaluate(
            lambda row: row.confidence >= threshold or row.margin <= limit,
            "either signal says head",
        ).describe()
    )


if __name__ == "__main__":
    main()
