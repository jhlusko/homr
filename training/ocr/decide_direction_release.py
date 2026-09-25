"""Apply the frozen B3 test gate to one selection-chosen detector checkpoint."""

import argparse
import json
import math
from pathlib import Path

from homr.text_detector_classes import DIRECTION_CLASS_ORDER


def counts(report: dict, corpus: str, label: str, role: str = "test") -> dict:
    return report["corpora"][corpus][role]["folded"][label]


def decide(
    parent: dict,
    e4: dict,
    selected: dict,
    candidate: dict,
    history: dict,
    direction_only: bool = False,
) -> dict:
    """`direction_only=True` is decision 13 (2026-09-25): this checkpoint ships
    `DirectionText` only. `Dynamic` comes from `e4` unchanged, so there is nothing new
    to test for it - the `ossq_dynamic_recall`/`ossq_dynamic_count` checks are dropped
    from the gate entirely rather than compared against a model that was never meant to
    supply that class. The synthetic per-class learning floor still includes `Dynamic`:
    that is a sanity check the head did not break during training, independent of
    whether production uses its output.
    """
    choice = selected["selected"]
    if choice is None:
        raise ValueError("no selection-eligible checkpoint; test must not be run")
    if candidate["weights_sha256"] != choice["weights_sha256"]:
        raise ValueError("test checkpoint differs from the selected checkpoint")
    if candidate.get("thresholds") != choice.get("thresholds"):
        raise ValueError("test must apply the selection-calibrated thresholds unchanged")
    if tuple(candidate["class_order"]) != DIRECTION_CLASS_ORDER:
        raise ValueError("test checkpoint has wrong class order")
    if set(candidate["corpora"]) != {"ossq_boxes", "lieder_boxes", "lieder_lyrics"}:
        raise ValueError("test corpus set changed")
    if any(set(roles) != {"test"} for roles in candidate["corpora"].values()):
        raise ValueError("candidate report must contain test pages only")
    if not (
        parent["split_manifest_sha256"]
        == e4["split_manifest_sha256"]
        == candidate["split_manifest_sha256"]
    ):
        raise ValueError("split manifest changed")
    pd = counts(parent, "ossq_boxes", "DirectionText")
    cd = counts(candidate, "ossq_boxes", "DirectionText")
    pl = counts(parent, "lieder_boxes", "DirectionText")
    cl = counts(candidate, "lieder_boxes", "DirectionText")
    py = counts(parent, "ossq_boxes", "Dynamic")
    ey = counts(e4, "ossq_boxes", "Dynamic")
    cy = counts(candidate, "ossq_boxes", "Dynamic")
    support_changed = (pd["ground_truth"], pl["ground_truth"]) != (
        cd["ground_truth"], cl["ground_truth"]
    )
    if not direction_only:
        support_changed = support_changed or (
            py["ground_truth"] != cy["ground_truth"] or ey["ground_truth"] != cy["ground_truth"]
        )
    if support_changed:
        raise ValueError("test support changed")
    valid = history["history"][choice["epoch"] - 1]["valid"]
    learned = all(
        valid.get(label, 0) >= .05 for label in ("Dynamic", "DirectionText", "Lyrics")
    ) and ("MeasureNumber" not in valid or valid["MeasureNumber"] >= .05)
    floors = {
        "ossq_direction_matches": math.ceil(pd["matched"] + .05 * pd["ground_truth"] - 1e-9),
        "lieder_direction_matches": pl["matched"],
    }
    caps = {
        "ossq_direction_predictions": pd["predicted"],
        "lieder_direction_predictions": pl["predicted"],
    }
    checks = {
        "ossq_direction_recall": cd["matched"] >= floors["ossq_direction_matches"],
        "lieder_direction_recall": cl["matched"] >= floors["lieder_direction_matches"],
        "ossq_direction_count": cd["predicted"] <= caps["ossq_direction_predictions"],
        "lieder_direction_count": cl["predicted"] <= caps["lieder_direction_predictions"],
        "synthetic_learning_floor": learned,
    }
    if not direction_only:
        floors["ossq_dynamic_matches"] = max(
            py["matched"], math.ceil(ey["matched"] - .05 * ey["ground_truth"] - 1e-9)
        )
        caps["ossq_dynamic_predictions"] = ey["predicted"]
        checks["ossq_dynamic_recall"] = cy["matched"] >= floors["ossq_dynamic_matches"]
        checks["ossq_dynamic_count"] = cy["predicted"] <= caps["ossq_dynamic_predictions"]
    return {
        "gate_pass": all(checks.values()), "checks": checks,
        "floors": floors, "prediction_caps": caps,
        "candidate": {"ossq_direction": cd, "lieder_direction": cl, "ossq_dynamic": cy},
        "selected_epoch": choice["epoch"], "weights_sha256": choice["weights_sha256"],
        "thresholds": choice.get("thresholds"),
        "split_manifest_sha256": candidate["split_manifest_sha256"],
        "direction_only": direction_only,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("parent", "e4", "selected", "candidate", "history", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    decision = decide(*[
        json.loads(getattr(args, name).read_text())
        for name in ("parent", "e4", "selected", "candidate", "history")
    ])
    args.out.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
