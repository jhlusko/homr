"""Bind reviewed OSSQ opening-scan boundaries to canonical source XML.

Source layout alone is not scan alignment. Explicit visual review and hash-bound
selection inputs are required; no recognizer output is consulted.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from training.onnx.qualify_joint_export import sha, write


def source_opening_windows(root):
    parts = root.findall("part")
    if root.tag != "score-partwise" or len(parts) != 4:
        raise ValueError("Expected four-part quartet source")
    if any("homr" in (s.text or "").lower() for s in root.findall(".//software")):
        raise ValueError("Recognizer output cannot be reference")
    windows, numbering, total = [], [], []
    for part in parts:
        measures = part.findall("measure")
        starts = [0]
        end = len(measures)
        for index, measure in enumerate(measures[1:], 1):
            mark = measure.find("print")
            if mark is not None and mark.get("new-page") == "yes":
                end = index
                break
            if mark is not None and mark.get("new-system") == "yes":
                starts.append(index)
        windows.append(list(zip(starts, starts[1:] + [end])))
        numbering.append([m.get("number") for m in measures[:end]])
        total.append(len(measures))
    if not numbering[0] or any(w != windows[0] for w in windows):
        raise ValueError("Part page/system boundaries disagree")
    if any(n != numbering[0] for n in numbering) or len(set(total)) != 1:
        raise ValueError("Part physical measure identities disagree")
    return windows[0], total[0]


def certify(selection_path, review_path, out):
    if out.exists():
        raise FileExistsError("Refusing to replace certified manifest")
    selection = json.loads(selection_path.read_text())
    review = json.loads(review_path.read_text())
    if review["selection_sha256"] != sha(selection_path):
        raise ValueError("Visual review is for a different selection")
    if not review.get("reviewer") or review.get("model_predictions_used") is not False:
        raise ValueError("Independent visual reviewer evidence required")
    pages = []
    for selected in selection:
        checked = review["scores"][selected["score"]]
        for flag in ("edition_identity_checked", "complete_page_checked", "part_order_checked"):
            if checked.get(flag) is not True:
                raise ValueError("Missing visual check: " + flag)
        reference, image = Path(selected["reference"]), Path(selected["image"])
        if sha(reference) != checked["reference_sha256"] or sha(image) != checked["image_sha256"]:
            raise ValueError("Reviewed source/page changed")
        root = ET.parse(reference).getroot()
        windows, total = source_opening_windows(root)
        counts = [end - start for start, end in windows]
        if counts != checked["visible_system_measure_counts"]:
            raise ValueError("Scan counts disagree with source layout")
        if (
            selected["proposed_start_measure"] != 0
            or selected["proposed_end_measure"] != windows[-1][1]
        ):
            raise ValueError("Selected page window changed")
        with Image.open(image) as pixels:
            dimensions = list(pixels.size)
        if dimensions != selected["dimensions"]:
            raise ValueError("Selected page dimensions changed")
        if dimensions[0] * dimensions[1] > selected["pixel_budget"]:
            raise ValueError("Page exceeds frozen pixel budget")
        pages.append(
            {
                **selected,
                "source": "ossq",
                "reference_sha256": sha(reference),
                "image_sha256": sha(image),
                "reference_total_measures": total,
                "start_measure": 0,
                "end_measure": windows[-1][1],
                "systems": [
                    {
                        "system_index": index,
                        "start_measure": start,
                        "end_measure": end,
                        "status": "aligned",
                        "boundary_certified": True,
                        "count_exact": True,
                        "certification": (
                            "Agent visual review; not exhaustive note-level reference audit"
                        ),
                    }
                    for index, (start, end) in enumerate(windows)
                ],
                "review": checked,
                "exposure_status": "Previously evaluated; historical model provenance incomplete",
            }
        )
    out.mkdir(parents=True)
    write(out / "pages.json", pages)
    write(
        out / "receipt.json",
        {
            "schema": "homr.ossq-opening-page-certification.v1",
            "tool_sha256": sha(__file__),
            "selection_sha256": sha(selection_path),
            "review_sha256": sha(review_path),
            "pages_sha256": sha(out / "pages.json"),
            "pages": len(pages),
            "model_predictions_used": False,
            "scope": (
                "Diagnostic page boundaries, not untouched-test or exhaustive annotation fidelity"
            ),
        },
    )
    print(json.dumps({"pages": len(pages), "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    certify(args.selection.resolve(), args.review.resolve(), args.out.resolve())
