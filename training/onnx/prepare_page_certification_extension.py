"""Freeze additional real-page diagnostic inputs before recognition or visual review."""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from training.onnx.certify_ossq_opening_pages import source_opening_windows
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference

SEED = "full-page-extension-20261005:"


def preview_window(root, start, end):
    """Slice a display-only source window, carrying prior staff-specific attributes."""
    result = copy.deepcopy(root)
    for part in result.findall("part"):
        measures = part.findall("measure")
        if not 0 <= start < end <= len(measures):
            raise ValueError("Invalid preview window")
        inherited = {}
        for measure in measures[:start]:
            for attrs in measure.findall("attributes"):
                for child in attrs:
                    if child.tag in ("divisions", "key", "time", "staves", "clef", "transpose"):
                        inherited[(child.tag, child.get("number"))] = copy.deepcopy(child)
        first = measures[start]
        existing = {
            (child.tag, child.get("number"))
            for attrs in first.findall("attributes")
            for child in attrs
        }
        carry = ET.Element("attributes")
        for key, child in inherited.items():
            if key not in existing:
                carry.append(child)
        if len(carry):
            first.insert(0, carry)
        for mark in first.findall("print"):
            mark.attrib.pop("new-page", None)
        for measure in measures[:start] + measures[end:]:
            part.remove(measure)
    return result


def prepare(workspace, out):
    if out.exists():
        raise FileExistsError("Refusing to replace frozen selection")
    base = workspace / "homr-artifacts/gpu-roadmap-20260919/instance-results/workspace"
    pool_path = base / "ossq-page-certification-20261004/eligible-candidates.json"
    audit_path = base / "full-page-reference-audit-20261003-final/lieder-page-audit.jsonl"
    alignment = workspace / "lieder-omr-data/provenance/alignment_v4_boundary_safe.json"
    if json.loads(alignment.read_text()).get("model_predictions_used") is not False:
        raise ValueError("Independent alignment required")
    pool = json.loads(pool_path.read_text())
    selected = []
    for candidate in pool["candidates"][2:4]:
        csv = Path(candidate["csv"])
        if sha(csv) != candidate["csv_sha256"]:
            raise ValueError("Candidate CSV changed")
        first = csv.read_text().splitlines()[0].split(":")[0]
        page_number = int(first) if first else 1
        image = csv.parent / f"images/scanned/original/{candidate['score']}:{page_number:04}.png"
        reference = csv.with_name(candidate["score"] + ".musicxml")
        windows, total = source_opening_windows(read_reference(reference))
        selected.append(
            {
                "score": candidate["score"],
                "source": "ossq",
                "image": str(image),
                "reference": str(reference),
                "proposed_start_measure": 0,
                "proposed_end_measure": windows[-1][1],
                "systems": [
                    {"start_measure": a, "end_measure": z, "system_index": i}
                    for i, (a, z) in enumerate(windows)
                ],
                "reference_total_measures": total,
                "known_exposure": candidate["known_index_exposure"],
                "exposure_status": (
                    "Published scanned test; previously evaluated; historical training "
                    "provenance incomplete"
                ),
                "selection": (
                    "Next two in the previously frozen OSSQ identity order; CSV "
                    "opening-page hint requires visual verification"
                ),
            }
        )
    groups = {}
    for line in audit_path.read_text().splitlines():
        row = json.loads(line)
        if (
            not row["image"]
            or not row["reference"]
            or not row["dimensions"]
            or any(r != "known_index_exposure" for r in row["exclusions"])
            or row["dimensions"][0] * row["dimensions"][1] > 10_000_000
            or row["score"] == "IMSLP323995"  # Lorelei already compared; retain its result.
            or any(
                ("public-core-reproducibility" in p and p.endswith("index_train.txt"))
                or ("public-joint-core-heads" in p and p.endswith("frozen-train-index.txt"))
                for p in row["known_exposure"]
            )
        ):
            continue
        groups.setdefault(row["score"], []).append(row)
    ordered = sorted(groups, key=lambda s: hashlib.sha256((SEED + s).encode()).hexdigest())
    if len(ordered) < 2:
        raise ValueError("Insufficient independent-source diagnostic candidates")
    for score in ordered[:2]:
        row = min(
            groups[score],
            key=lambda r: hashlib.sha256(Path(r["image"]).name.encode()).hexdigest(),
        )
        selected.append(
            {
                **row,
                "proposed_start_measure": min(r["start_measure"] for r in row["systems"]),
                "proposed_end_measure": max(r["end_measure"] for r in row["systems"]),
                "exposure_status": (
                    "Known validation/historical index exposure; diagnostic only, not "
                    "untouched test"
                ),
                "selection": (
                    "Identity hash order, excluding public-core/joint training-index "
                    "exposure and already-compared Lorelei"
                ),
            }
        )
    out.mkdir(parents=True)
    jobs = []
    for row in selected:
        with Image.open(row["image"]) as image:
            row["dimensions"] = list(image.size)
        row["pixel_budget"] = 10_000_000
        if row["dimensions"][0] * row["dimensions"][1] > row["pixel_budget"]:
            raise ValueError("Selected page exceeds fixed budget")
        row["image_sha256"] = sha(row["image"])
        row["reference_sha256"] = sha(row["reference"])
        folder = out / row["score"]
        folder.mkdir()
        preview = preview_window(
            read_reference(row["reference"]),
            row["proposed_start_measure"],
            row["proposed_end_measure"],
        )
        path = folder / "source-preview.musicxml"
        ET.ElementTree(preview).write(path, encoding="utf-8", xml_declaration=True)
        row["preview_musicxml"] = str(path)
        row["preview_musicxml_sha256"] = sha(path)
        jobs.append({"in": str(path), "out": str(folder / "source-preview.pdf")})
    write(out / "selection.json", selected)
    write(out / "render-jobs.json", jobs)
    write(
        out / "selection-receipt.json",
        {
            "seed": SEED,
            "tool_sha256": sha(__file__),
            "selection_sha256": sha(out / "selection.json"),
            "inputs": {str(p): sha(p) for p in (pool_path, audit_path, alignment)},
            "lieder_candidate_scores": ordered,
            "model_predictions_used": False,
            "scope": (
                "Proposed diagnostic inputs; visual certification still required; "
                "previews are not ground truth"
            ),
        },
    )
    print(json.dumps({"selected": [r["score"] for r in selected]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.out.resolve())
