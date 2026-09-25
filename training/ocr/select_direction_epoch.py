"""Select a detector epoch by the frozen B3 real-page recall rule, without test access."""

import argparse
import json
import math
from pathlib import Path


def class_counts(report: dict, corpus: str, label: str, candidate: bool = False) -> dict:
    if candidate and set(report["corpora"][corpus]) != {"selection"}:
        raise ValueError("epoch report must contain selection only; test access is forbidden")
    return report["corpora"][corpus]["selection"]["folded"][label]


def select(parent: dict, candidates: list[dict], history: dict) -> dict:
    base_direction = class_counts(parent, "ossq_boxes", "DirectionText")
    base_dynamic = class_counts(parent, "ossq_boxes", "Dynamic")
    base_lieder = class_counts(parent, "lieder_boxes", "DirectionText")
    floors = {
        "ossq_direction": math.ceil(base_direction["matched"] + 0.03 * base_direction["ground_truth"] - 1e-9),
        "ossq_dynamic": math.ceil(base_dynamic["matched"] - 0.02 * base_dynamic["ground_truth"] - 1e-9),
        "lieder_direction": base_lieder["matched"] - 1,
    }
    ranked = []
    for epoch, report in enumerate(candidates, start=1):
        if report["split_manifest_sha256"] != parent["split_manifest_sha256"]:
            raise ValueError("split manifest changed between baseline and epoch")
        ossq = class_counts(report, "ossq_boxes", "DirectionText", candidate=True)
        dynamic = class_counts(report, "ossq_boxes", "Dynamic", candidate=True)
        lieder = class_counts(report, "lieder_boxes", "DirectionText", candidate=True)
        if (
            ossq["ground_truth"] != base_direction["ground_truth"]
            or dynamic["ground_truth"] != base_dynamic["ground_truth"]
            or lieder["ground_truth"] != base_lieder["ground_truth"]
        ):
            raise ValueError("selection support changed")
        validation = history["history"][epoch - 1]["valid"]
        learned = all(validation.get(label, 0) >= 0.05 for label in ("DirectionText", "Dynamic", "Lyrics"))
        eligible = (
            ossq["matched"] >= floors["ossq_direction"]
            and dynamic["matched"] >= floors["ossq_dynamic"]
            and lieder["matched"] >= floors["lieder_direction"]
            and learned
        )
        ranked.append(
            {
                "epoch": epoch,
                "weights": report["weights"],
                "weights_sha256": report["weights_sha256"],
                "ossq_direction": ossq,
                "ossq_dynamic": dynamic,
                "lieder_direction": lieder,
                "synthetic_gate_ok": learned,
                "eligible": eligible,
            }
        )
    eligible = [row for row in ranked if row["eligible"]]
    best = max(
        eligible,
        key=lambda row: (
            row["ossq_direction"]["matched"], row["ossq_dynamic"]["matched"], -row["epoch"]
        ),
        default=None,
    )
    return {"criterion": "frozen B3 selection recall floors; test not accessed", "floors": floors,
            "selected": best, "candidates": ranked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--epoch", type=Path, nargs="+", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = select(
        json.loads(args.parent.read_text()),
        [json.loads(path.read_text()) for path in args.epoch],
        json.loads(args.history.read_text()),
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"floors": result["floors"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
