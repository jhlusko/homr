"""Inspect predicted peer-bar length support without consulting source references.

This is a coverage diagnostic, not permission to infer a time signature or edit XML.
Only read trusted locally generated pickle files.
"""

import argparse
import json
import pickle
from collections import Counter
from fractions import Fraction
from pathlib import Path

from training.onnx.probe_selective_timing_export import bars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.run / "completion.json").read_text())
    results = []
    for path in sorted(args.run.glob("*/observed/final-export-input.pkl")):
        with path.open("rb") as handle:
            _, voices, _ = pickle.load(handle)
        parts = []
        for pi, voice in enumerate(voices):
            grouped = bars(voice)
            single_staff = not any(s.position == "lower" for s in voice)
            lengths = [
                sum((g.get_render_duration() * 4 for g in bar), Fraction())
                for bar in grouped
                if any(
                    s.rhythm.startswith("note") and "G" not in s.rhythm
                    for g in bar
                    for s in g.symbols
                )
            ]
            counts = Counter(lengths)
            modal, support = counts.most_common(1)[0] if counts else (None, 0)
            parts.append(
                dict(
                    part=pi + 1,
                    bar_count=len(grouped),
                    single_staff=single_staff,
                    declared_tokens=[
                        s.rhythm
                        for s in voice
                        if s.rhythm.startswith(("timeSignature/", "timeSignatureBeats_"))
                    ],
                    bar_length_votes={str(k): v for k, v in sorted(counts.items())},
                    modal_quarters=str(modal) if modal is not None else None,
                    modal_support=support,
                    total=len(lengths),
                    meets_coarse_support=single_staff
                    and len(lengths) >= 3
                    and support * 4 >= len(lengths) * 3,
                )
            )
        results.append(dict(score=path.parent.parent.name, parts=parts))
    output = dict(
        limitation="Predicted-only coarse coverage audit. Candidate bar is not excluded "
        "from these descriptive totals; any future proposal must exclude it, require "
        "aligned bar counts and validate timelines. No meter inference or XML changes.",
        completed_job_count=len(receipt.get("jobs", [])),
        cases=results,
    )
    (args.run / "peer-context-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in results:
        print(
            case["score"],
            [
                (
                    p["part"],
                    p["bar_count"],
                    p["modal_quarters"],
                    p["modal_support"],
                    p["total"],
                    p["meets_coarse_support"],
                )
                for p in case["parts"]
            ],
        )


if __name__ == "__main__":
    main()
