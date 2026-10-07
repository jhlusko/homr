"""Freeze score-disjoint positive pages for the fixed research tremolo rule."""

import argparse
import json
from pathlib import Path

from training.onnx.probe_staff_lane_timing import certified_window
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference

SELECTED = {
    "IMSLP589008": "IMSLP589008-p008.png",
    "IMSLP589118": "IMSLP589118-p007.png",
    "IMSLP634835": "IMSLP634835-p004.png",
    "IMSLP83318": "IMSLP83318-p006.png",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = args.base / "full-page-reference-audit-20261003-final/lieder-page-audit.jsonl"
    ossq = args.base / "ossq-page-certification-20261004/certified/pages.json"
    pages = []
    for line in audit.read_text().splitlines():
        row = json.loads(line)
        if not row["image"] or Path(row["image"]).name != SELECTED.get(row["score"]):
            continue
        page = dict(
            row,
            start_measure=row["systems"][0]["start_measure"],
            end_measure=row["systems"][-1]["end_measure"],
            image_sha256=sha(row["image"]),
            reference_sha256=sha(row["reference"]),
            exposure_status="Known historical exposure; score-disjoint from rule development, not untouched test",
        )
        pages.append(page)
    pages += json.loads(ossq.read_text())
    if len(pages) != 6 or len({r["score"] for r in pages}) != 6:
        raise ValueError("Require six distinct fixed scores")
    semantics = []
    for page in pages:
        if not certified_window(page) or page["dimensions"][0] * page["dimensions"][1] > 14_000_000:
            raise ValueError("Uncertified or oversized page")
        page["boundary_certified"] = True
        root = read_reference(page["reference"])
        notes = [
            n
            for p in root.findall("part")
            for m in p.findall("measure")[page["start_measure"] : page["end_measure"]]
            for n in m.findall("note")
        ]
        tremolos = [n.find("notations/ornaments/tremolo") for n in notes]
        paired = sum(t is not None and t.get("type") in ("start", "stop") for t in tremolos)
        single = sum(t is not None and t.get("type", "single") == "single" for t in tremolos)
        semantics.append(
            dict(
                score=page["score"],
                paired_tremolo_carriers=paired,
                single_tremolo_carriers=single,
                note_support=len(notes),
            )
        )
        if paired == 0 and page["source"] != "ossq":
            raise ValueError(
                "Positive cohort lacks paired-tremolo source support: " + page["score"]
            )
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "pages.json", pages)
    write(
        args.out / "selection-receipt.json",
        dict(
            source_sha256=sha(__file__),
            input_sha256={str(p): sha(p) for p in (audit, ossq)},
            source_semantics=semantics,
            policy_sha256=sha(Path(__file__).with_name("probe_staff_lane_timing.py")),
            selection="Four highest-count distinct available certified Lieder paired-tremolo scores, plus two already-certified OSSQ tremolo pages (classify paired/single in the receipt). Frozen before new inference; source-positive selection only, not predicted gain selection.",
            limitations="Policy-development-score-disjoint; earlier recognition/model exposure remains. Previously inspected controls retained separately.",
        ),
    )


if __name__ == "__main__":
    main()
