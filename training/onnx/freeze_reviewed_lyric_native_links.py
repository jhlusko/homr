"""Freeze approved scan/native passage links without claiming syllable pixel truth."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(review_dir, out):
    review_dir, out = Path(review_dir), Path(out)
    report_path = review_dir / "report.json"
    approval_path = review_dir / "owner-review-whole-set.json"
    inspection_path = review_dir / "visual-inspection-complete.json"
    report = json.loads(report_path.read_text())
    approval = json.loads(approval_path.read_text())
    inspection = json.loads(inspection_path.read_text())
    if approval.get("wholeSetApproved") is not True or approval.get("reportSha256") != sha(
        report_path
    ):
        raise ValueError("Whole-set approval must bind this exact report")
    if inspection.get("reportSha256") != sha(report_path) or inspection.get(
        "ownerApprovalSha256"
    ) != sha(approval_path):
        raise ValueError("Inspection/approval evidence changed")
    if report.get("model_predictions_used") is not False:
        raise ValueError("Prediction-assisted reference links refused")
    for field, file in [("manualContextSha256", "manual-system-context.json")]:
        if inspection.get(field) != sha(review_dir / file):
            raise ValueError("Manual context evidence changed")
    checked = {(a["score"], a["system"]): a for a in inspection["systems"]}
    if len(checked) != len(inspection["systems"]):
        raise ValueError("Duplicate system inspection")
    pages, expected = [], set()
    for page in report["pages"]:
        if (
            sha(page["image"]) != page["image_sha256"]
            or sha(page["native_source"]) != page["native_source_sha256"]
        ):
            raise ValueError("Frozen scan or native source changed")
        cursor = page["start_measure"]
        for s in page["systems"]:
            key = (page["score"], s["index"] + 1)
            expected.add(key)
            a = checked.get(key, {})
            if not a.get("status", "").startswith("passage_and_boundary_visual_checked"):
                raise ValueError("Missing complete system passage inspection")
            if s["start_measure"] != cursor or s["end_measure"] <= cursor:
                raise ValueError("Noncontiguous native measure coverage")
            if (a["nativeStart"], a["nativeEnd"]) != (s["start_measure"] + 1, s["end_measure"]):
                raise ValueError("Inspected native measure window differs")
            contact = review_dir / "inspection" / f"{page['score']}-system{s['index']+1}.png"
            if sha(contact) != a["contactSha256"]:
                raise ValueError("Inspected contact pixels changed")
            cursor = s["end_measure"]
        if cursor != page["end_measure"]:
            raise ValueError("Incomplete page measure coverage")
        pages.append(
            {
                **page,
                "native_passage_link_approved": True,
                "pixel_link_certified": False,
                "certification_scope": (
                    "Approved native passage/system boundaries and part correspondence; "
                    "not exhaustive annotation fidelity, exact syllable pixels, "
                    "or untouched-test independence."
                ),
            }
        )
    if set(checked) != expected:
        raise ValueError("Inspection inventory differs from selected systems")
    if out.exists():
        raise FileExistsError("Refusing to replace frozen reference links")
    out.mkdir(parents=True)
    (out / "pages.json").write_text(json.dumps(pages, indent=2) + "\n")
    receipt = {
        "schema": "ots.approved-native-lyric-links.v1",
        "pages": len(pages),
        "systems": len(expected),
        "nativeTargets": sum(p["page_native_lyric_targets"] for p in pages),
        "textExceptions": inspection["textExceptions"],
        "model_predictions_used": False,
        "pixel_link_certified": False,
        "release_gate_passed": False,
        "inputSha256": {str(p): sha(p) for p in (report_path, approval_path, inspection_path)},
        "pagesSha256": sha(out / "pages.json"),
        "sourceSha256": sha(__file__),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.review_dir, args.out)))
