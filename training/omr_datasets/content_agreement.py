"""How far each label's NOTES diverge from the model's reading of the same crop.

Every gate in this pipeline checks measure RANGES. Nothing checks content, and human
review found the difference: two evaluation pairs marked "correct line, with errors" -
the range is right and the notes inside it are not. Accuracy is measured against those
labels.

This measures the gap. It deliberately does **not** close it, because the only content
signal available is the model's own reading, and filtering an evaluation set by
agreement with the model is precisely the circularity that condemned
`recover_excluded_pairs`: labels the model already reproduces survive, labels it gets
wrong are discarded, and the resulting accuracy measures the model against itself.

So the output is a ranking for human attention, never a filter. Ordering review by
content disagreement spends a reviewer's time where content is most likely wrong,
without letting the model decide what counts as truth.

For the TRAINING set the same objection does not apply with equal force - pseudo-labels
are already model-derived and tagged as such - but that is a separate decision, and
this module does not make it either.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

PAD = "\x00missing"


def content_agreement(record: dict) -> tuple[float, int]:
    """`(agreement, notes compared)` on pitch AND rhythm together.

    Both branches must match for a position to count as agreeing: the same pitch with
    the wrong duration is a content error, and so is the reverse. Padded positions are
    excluded on either side - they mark a length disagreement, which is a RANGE
    question and already has its own gates.
    """
    ref_pitch, got_pitch = record["pitch_reference"], record["pitch_predicted"]
    ref_rhythm, got_rhythm = record["rhythm_reference"], record["rhythm_predicted"]
    real = [
        i for i, (a, b) in enumerate(zip(ref_rhythm, got_rhythm))
        if not a.startswith(PAD) and not b.startswith(PAD)
    ]
    if not real:
        return 1.0, 0
    hit = sum(
        1 for i in real
        if ref_pitch[i] == got_pitch[i] and ref_rhythm[i] == got_rhythm[i]
    )
    return hit / len(real), len(real)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-notes", type=int, default=8,
                        help="Ignore staves too short for a rate to mean anything.")
    args = parser.parse_args()

    rows = []
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        agreement, notes = content_agreement(record)
        if notes >= args.min_notes:
            rows.append({"stem": Path(record["tokens"]).stem,
                         "agreement": round(agreement, 4), "notes": notes})
    rows.sort(key=lambda r: r["agreement"])

    values = [r["agreement"] for r in rows]
    summary = {
        "staves": len(rows),
        "median": values[len(values) // 2] if values else None,
        "tenth_percentile": values[len(values) // 10] if values else None,
        "below_0.9": sum(1 for v in values if v < 0.9),
        "below_0.5": sum(1 for v in values if v < 0.5),
        "circularity_warning": (
            "A ranking for human attention, not a filter. Selecting evaluation labels "
            "by agreement with the model measures the model against itself."
        ),
        "rows": rows,
    }
    args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"staves: {summary['staves']}")
    print(f"  median agreement:  {summary['median']}")
    print(f"  10th percentile:   {summary['tenth_percentile']}")
    print(f"  below 0.9:         {summary['below_0.9']}")
    print(f"  below 0.5:         {summary['below_0.5']}")


if __name__ == "__main__":
    main()
