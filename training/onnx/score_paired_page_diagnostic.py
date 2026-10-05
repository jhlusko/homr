"""Score a matched diagnostic page; preserve decode failures as failures."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_full_page_musicxml import compare, staff_events


def read_reference(path):
    path = Path(path)
    root = read_mxl(path) if path.suffix.lower() == ".mxl" else ET.parse(path).getroot()
    if root.tag != "score-partwise" or any(
        "homr" in (s.text or "").lower() for s in root.findall(".//software")
    ):
        raise ValueError("Unsupported or recognizer-generated reference")
    return root


def score_pair(page, directory):
    if sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Reference changed")
    reference = staff_events(
        read_reference(page["reference"]), page["start_measure"], page["end_measure"]
    )
    arms, identities = {}, []
    for arm in ("accepted", "candidate"):
        folder = directory / arm
        identity = json.loads((folder / "identity.json").read_text())
        if identity["page"] != page or identity["arm"] != arm:
            raise ValueError("Page/arm identity mismatch")
        identities.append(identity)
        receipt = json.loads((folder / "result.json").read_text())
        if receipt["status"] != "ok":
            arms[arm] = {
                "status": "failed",
                "seconds": receipt["seconds"],
                "error": receipt["error"],
            }
            continue
        xml = folder / "page.musicxml"
        if sha(xml) != receipt["xml_sha256"]:
            raise ValueError("Prediction changed")
        arms[arm] = {
            "status": "ok",
            "seconds": receipt["seconds"],
            "metrics": compare(reference, staff_events(ET.parse(xml).getroot())),
            "xml_sha256": receipt["xml_sha256"],
            "counts": receipt["counts"],
        }
    for field in ("policy", "code_sha256", "segmentation", "threads", "device"):
        if identities[0][field] != identities[1][field]:
            raise ValueError("Unmatched runtime protocol: " + field)
    report = {
        "page": page,
        "arms": arms,
        "successful_arms": sum(a["status"] == "ok" for a in arms.values()),
        "attempted_arms": 2,
        "scorer_sha256": sha(__file__),
        "limitations": (
            "Diagnostic page with recorded exposure provenance; "
            "not untouched test or crop token SER. "
            "Ordered fixed-staff event edits omit onset/barline/notation/lyrics gates. "
            "Float32 local CPU path excludes text/OCR/provider overhead."
        ),
    }
    write(directory / "comparison.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--page-index", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = score_pair(json.loads(args.pages.read_text())[args.page_index], args.run)
    print(json.dumps(report, indent=2), flush=True)
