"""Certify hash-bound reviewed real-page windows without using recognizer outputs."""

import argparse
import json
from pathlib import Path

from training.onnx.audit_full_page_references import certified_page
from training.onnx.certify_ossq_opening_pages import source_opening_windows
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference


def certify(selection_path, review_path, out):
    if out.exists():
        raise FileExistsError("Refusing to replace certification")
    selected = json.loads(selection_path.read_text())
    review = json.loads(review_path.read_text())
    receipt_path = selection_path.with_name("selection-receipt.json")
    receipt = json.loads(receipt_path.read_text())
    if sha(selection_path) != receipt["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    for path, digest in receipt["inputs"].items():
        if sha(path) != digest:
            raise ValueError("Selection evidence changed")
    if review["selection_sha256"] != sha(selection_path):
        raise ValueError("Review covers a different selection")
    if not review.get("reviewer") or review.get("model_predictions_used") is not False:
        raise ValueError("Independent reviewer required")
    pages = []
    for row in selected:
        checked = review["scores"][row["score"]]
        for flag in ("edition_identity_checked", "complete_page_checked", "part_order_checked"):
            if checked.get(flag) is not True:
                raise ValueError("Missing visual review: " + flag)
        for name in ("image", "reference"):
            if sha(row[name]) != row[name + "_sha256"]:
                raise ValueError("Selected input changed")
            if checked[name + "_sha256"] != row[name + "_sha256"]:
                raise ValueError("Reviewed input changed")
        root = read_reference(row["reference"])
        parts = root.findall("part")
        counts = [len(p.findall("measure")) for p in parts]
        if len(set(counts)) != 1 or counts[0] != row["reference_total_measures"]:
            raise ValueError("Native part physical measure counts disagree")
        systems = row["systems"]
        start, end = row["proposed_start_measure"], row["proposed_end_measure"]
        if row["source"] == "ossq":
            windows, _ = source_opening_windows(root)
            if windows != [(r["start_measure"], r["end_measure"]) for r in systems]:
                raise ValueError("Native OSSQ source layout changed")
        elif row["source"] == "lieder":
            if not certified_page(systems):
                raise ValueError("Unqualified independent Lieder page boundaries")
        else:
            raise ValueError("Unsupported source")
        if (
            start != systems[0]["start_measure"]
            or end != systems[-1]["end_measure"]
            or not 0 <= start < end <= counts[0]
            or any(a["end_measure"] != z["start_measure"] for a, z in zip(systems, systems[1:]))
        ):
            raise ValueError("Page window is not contiguous")
        if checked["visible_system_measure_counts"] != [
            r["end_measure"] - r["start_measure"] for r in systems
        ]:
            raise ValueError("Visual system counts disagree")
        if len(checked["boundary_evidence"]) != len(systems):
            raise ValueError("Missing per-system boundary evidence")
        for relative, digest in checked["review_artifacts"].items():
            if sha(selection_path.parent / relative) != digest:
                raise ValueError("Review artifact changed")
        pages.append(
            {
                **row,
                "start_measure": start,
                "end_measure": end,
                "systems": [
                    {**r, "status": "aligned", "boundary_certified": True, "count_exact": True}
                    for r in systems
                ],
                "review": checked,
                "certification_scope": (
                    "Agent visual page/edition/part/boundary review; not exhaustive "
                    "note-level fidelity or untouched-test certificate"
                ),
            }
        )
    out.mkdir(parents=True)
    write(out / "pages.json", pages)
    write(
        out / "receipt.json",
        {
            "schema": "homr.real-page-extension-certification.v1",
            "tool_sha256": sha(__file__),
            "selection_sha256": sha(selection_path),
            "review_sha256": sha(review_path),
            "pages_sha256": sha(out / "pages.json"),
            "pages": len(pages),
            "model_predictions_used": False,
            "scope": "Additional diagnostic windows; no new untouched-test claim",
        },
    )
    print(json.dumps({"certified": len(pages)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "review", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    certify(args.selection.resolve(), args.review.resolve(), args.out.resolve())
