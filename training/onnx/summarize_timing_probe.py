"""Summarize completed teacher-forced timing curves without promotion claims."""

import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-support", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    history_path = args.run / "full/history.json"
    history = json.loads(history_path.read_text())
    support = json.loads(args.training_support.read_text())["sources"]
    latest = history[-1]
    result = {}
    lines = [
        "# Frozen-core timing probe: teacher-forced results",
        "",
        "Calibration/validation representation probe only. No free decoding, runtime activation or promotion.",
        "",
        "| Source | Head | Events | Accuracy | Constant baseline | Macro recall | Best loss epoch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, heads in latest["validation"].items():
        result[source] = {}
        for head, metric in heads.items():
            training = support[source][head]
            majority = max(training, key=lambda label: (training[label], label))
            eligible = metric["eligible"]
            classes = {
                label: dict(
                    support=count,
                    correct=metric["correct"].get(label, 0),
                    recall=metric["correct"].get(label, 0) / count,
                )
                for label, count in metric["support"].items()
                if count
            }
            macro = sum(c["recall"] for c in classes.values()) / len(classes) if classes else None
            baseline = metric["support"].get(majority, 0) / eligible if eligible else None
            best = min(history, key=lambda epoch: epoch["validation"][source][head]["loss"])
            result[source][head] = dict(
                epoch=latest["epoch"],
                support=eligible,
                accuracy=metric["accuracy"],
                loss=metric["loss"],
                constant_class=majority,
                constant_accuracy=baseline,
                macro_recall=macro,
                classes=classes,
                minimum_validation_loss_epoch=best["epoch"],
                loss_by_epoch=[
                    dict(epoch=e["epoch"], loss=e["validation"][source][head]["loss"])
                    for e in history
                ],
            )
            lines.append(
                f"| {source} | {head} | {eligible} | {metric['accuracy']:.2%} | "
                f"{baseline:.2%} | {macro:.2%} | {best['epoch']} |"
                if eligible
                else f"| {source} | {head} | 0 | — | — | — | — |"
            )
    lines += [
        "",
        "Read OSSQ support and full-rest positive coverage in report.json; small samples cannot establish generalization.",
        "Lieder minority classes must be read separately; overall accuracy is dominated by ordinary notes.",
        "Epoch selection by these curves is exploratory and needs independent free-decoding qualification.",
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "report.json",
        dict(
            sources=result,
            final_epoch=latest["epoch"],
            inputs_sha256={
                str(p): sha(p)
                for p in [
                    history_path,
                    args.training_support,
                    args.run / "dataset.json",
                    args.run / f"full/epoch-{latest['epoch']}.pth",
                ]
            },
            code_sha256=sha(Path(__file__)),
            limitation="Teacher-forced validation only; not free decoding. "
            "OSSQ two-score validation; no positive full-rest targets. Inference still rejects probe-trained heads.",
        ),
    )
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
