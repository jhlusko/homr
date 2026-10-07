"""Freeze diagnostic pages containing previously inspected timing positives."""

import argparse
import copy
import json
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write

SELECTION = {
    "IMSLP154110": "IMSLP154110-p001.png",
    "IMSLP578231": "IMSLP578231-p002.png",
    "IMSLP621832": "IMSLP621832-p003.png",
    "IMSLP16369": "IMSLP16369-p004.png",
}


def certify_leading_system(systems):
    rows = copy.deepcopy(systems)
    if len(rows) < 2 or rows[0]["status"] != "ambiguous":
        return rows, None
    first, anchor = rows[:2]
    count = first.get("observed_measures")
    if not (
        first.get("boundary_certified")
        and first.get("count_exact")
        and isinstance(count, int)
        and count > 0
        and count == first.get("detected_measures")
        and anchor["status"] == "aligned"
        and anchor.get("boundary_certified")
        and anchor.get("count_exact")
        and first["system_index"] + 1 == anchor["system_index"]
        and first.get("scan_index", -2) + 1 == anchor.get("scan_index", -1)
        and anchor["start_measure"] >= count
        and first["end_measure"] == anchor["start_measure"]
        and first["start_measure"] == anchor["start_measure"] - count
    ):
        return rows, None
    first["status"] = "aligned"
    first["reason"] = "Certified leading count anchored to immediately following aligned system"
    return rows, dict(
        method="next-system-anchor-minus-certified-native-bar-count",
        source_anchor_measure=anchor["start_measure"],
        leading_bars=count,
        original_status="ambiguous",
        model_predictions_used=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.base.parents[3]
    alignment = workspace / "lieder-omr-data/provenance/alignment_v4_boundary_safe.json"
    if json.loads(alignment.read_text()).get("model_predictions_used") is not False:
        raise ValueError("Require source-independent alignment")
    audit = args.base / "full-page-reference-audit-20261003-final/lieder-page-audit.jsonl"
    selected = []
    for line in audit.read_text().splitlines():
        row = json.loads(line)
        if not row["image"] or Path(row["image"]).name != SELECTION.get(row["score"]):
            continue
        systems, derivation = certify_leading_system(row["systems"])
        certified = all(
            s["status"] == "aligned" and s.get("boundary_certified") and s.get("count_exact")
            for s in systems
        )
        certified &= all(
            a["end_measure"] == b["start_measure"] for a, b in zip(systems, systems[1:])
        )
        selected.append(
            dict(
                score=row["score"],
                source="lieder",
                image=row["image"],
                reference=row["reference"],
                image_sha256=sha(row["image"]),
                reference_sha256=sha(row["reference"]),
                start_measure=systems[0]["start_measure"],
                end_measure=systems[-1]["end_measure"],
                systems=systems,
                dimensions=row["dimensions"],
                boundary_certified=bool(certified),
                boundary_derivation=derivation,
                original_systems=row["systems"],
                exposure_status="Known validation/historical exposure; selected after crop results; diagnostic only",
                known_exposure=row["known_exposure"],
                selection="Known full-rest flag positives, plus dominant duplet-positive page from inspected crop benchmark",
            )
        )
    if len(selected) != len(SELECTION):
        raise ValueError("Missing selected pages")
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "pages.json", selected)
    write(
        args.out / "receipt.json",
        dict(
            audit_sha256=sha(audit),
            alignment_sha256=sha(alignment),
            source_sha256=sha(__file__),
            model_predictions_used_for_boundaries=False,
            pixel_budget="Diagnostic expanded to 14 million original pixels; no resizing",
            certified=sum(r["boundary_certified"] for r in selected),
        ),
    )


if __name__ == "__main__":
    main()
