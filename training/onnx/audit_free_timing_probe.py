"""Report paired passive-head gains/regressions and class support after free decoding."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def audit(report):
    paired = defaultdict(lambda: defaultdict(Counter))
    classes = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    changes = []
    for case in report["cases"]:
        events = defaultdict(dict)
        for observation in case["timing"]:
            events[(observation["reference_token_index"], observation["head"])][
                observation["arm"]
            ] = observation
        for (index, head), arms in events.items():
            baseline = arms["legacy"]
            for arm, value in arms.items():
                group = classes[case["source"]][arm + "/" + head][value["target"]]
                matched = value["predicted_token_index"] is not None
                correct = value["prediction"] == value["target"]
                group.update(support=1, matched=int(matched), correct=int(correct))
                if arm == "legacy":
                    continue
                counters = paired[case["source"]][arm + "/" + head]
                counters.update(support=1)
                if not matched:
                    counters.update(unmatched=1)
                    continue
                base_correct = baseline["prediction"] == baseline["target"]
                outcome = (
                    "both_correct"
                    if base_correct and correct
                    else (
                        "improvement" if correct else "regression" if base_correct else "both_wrong"
                    )
                )
                counters.update({outcome: 1})
                if outcome in ("improvement", "regression"):
                    changes.append(
                        dict(
                            id=case["id"],
                            source=case["source"],
                            score=case["score"],
                            reference_token_index=index,
                            predicted_token_index=value["predicted_token_index"],
                            head=head,
                            arm=arm,
                            outcome=outcome,
                            target=value["target"],
                            legacy=baseline["prediction"],
                            candidate=value["prediction"],
                        )
                    )
    return dict(
        paired=paired,
        classes=classes,
        changes=changes,
        limitation="Paired labels on strictly aligned measures only. Unmatched targets remain visible. "
        "No change to greedy recognition, runtime arbitration or MusicXML; not whole-score improvement.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()))
    result.update(report_sha256=sha(args.report), code_sha256=sha(__file__))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "paired-report.json", result)
    lines = [
        "# Image-only timing: paired label changes",
        "",
        result["limitation"],
        "",
        "| Source | Head / epoch | Improvements | Regressions | Unmatched |",
        "|---|---|---:|---:|---:|",
    ]
    for source, heads in result["paired"].items():
        for head, counts in heads.items():
            lines.append(
                f"| {source} | {head} | {counts['improvement']} | {counts['regression']} | {counts['unmatched']} |"
            )
    (args.out / "paired-report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
