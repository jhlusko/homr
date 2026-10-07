"""Freeze additional source-stratified controls before fixed-candidate inference."""

import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from training.onnx.probe_staff_lane_timing import certified_window
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference

EXCLUDED = {
    "IMSLP16369",
    "IMSLP589008",
    "IMSLP589118",
    "IMSLP634835",
    "IMSLP83318",
    "IMSLP154086",
    "IMSLP89797",
    "IMSLP154110",
    "IMSLP578231",
    "IMSLP621832",
}


@lru_cache(maxsize=512)
def reference(path):
    return read_reference(path)


def semantics(page):
    root = reference(page["reference"])
    counts = Counter(dict.fromkeys(("notes", "paired", "single", "grace", "triplet", "chord"), 0))
    meters = set()
    for part in root.findall("part"):
        for measure in part.findall("measure")[page["start_measure"] : page["end_measure"]]:
            for time in measure.findall("attributes/time"):
                meters.add((time.findtext("beats"), time.findtext("beat-type")))
            notes = measure.findall("note")
            counts["notes"] += len(notes)
            for note in notes:
                tremolo = note.find("notations/ornaments/tremolo")
                if tremolo is not None:
                    counts["paired" if tremolo.get("type") in ("start", "stop") else "single"] += 1
                counts["grace"] += note.find("grace") is not None
                counts["triplet"] += note.findtext("time-modification/actual-notes") == "3"
                counts["chord"] += note.find("chord") is not None
    return dict(counts, meter_declarations=sorted(meters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = args.base / "full-page-reference-audit-20261003-final/lieder-page-audit.jsonl"
    eligible = []
    for line in source.read_text().splitlines():
        row = json.loads(line)
        if row["score"] in EXCLUDED or not row.get("image") or not Path(row["image"]).exists():
            continue
        systems = row.get("systems", [])
        if (
            not systems
            or not row.get("dimensions")
            or any("start_measure" not in s or "end_measure" not in s for s in systems)
            or row["dimensions"][0] * row["dimensions"][1] > 14_000_000
        ):
            continue
        page = dict(
            row, start_measure=systems[0]["start_measure"], end_measure=systems[-1]["end_measure"]
        )
        if not certified_window(page):
            continue
        stats = semantics(page)
        if stats["paired"]:
            continue
        eligible.append((page, stats))
    # Source semantics, never predicted gains, decide the fixed negative cohort.
    categories = [
        ("changing-meter", lambda s: len(s["meter_declarations"]) > 1),
        ("single-tremolo", lambda s: s["single"] > 0),
        ("grace-and-triplets", lambda s: s["grace"] > 0 and s["triplet"] > 0),
        ("ordinary-chords", lambda s: s["single"] == 0 and s["chord"] > 0),
    ]
    selected, receipts, used = [], [], set()
    for category, predicate in categories:
        candidates = [(p, s) for p, s in eligible if p["score"] not in used and predicate(s)]
        if not candidates:
            raise ValueError("No certified local control for " + category)
        page, stats = min(
            candidates,
            key=lambda x: (
                x[0]["dimensions"][0] * x[0]["dimensions"][1],
                x[0]["score"],
                x[0]["image"],
            ),
        )
        page.update(
            boundary_certified=True,
            image_sha256=sha(page["image"]),
            reference_sha256=sha(page["reference"]),
            exposure_status=(
                "Historical model exposure tracked; " "disjoint from context-development scores"
            ),
        )
        selected.append(page)
        used.add(page["score"])
        receipts.append(dict(category=category, score=page["score"], image=page["image"], **stats))
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "pages.json", selected)
    write(
        args.out / "selection-receipt.json",
        {
            "source_sha256": sha(__file__),
            "input_sha256": sha(source),
            "selected": receipts,
            "policy_sha256": sha(Path(__file__).with_name("probe_staff_lane_timing.py")),
            "peer_span_sha256": sha(Path(__file__).with_name("tremolo_peer_span.py")),
            "selection": (
                "One smallest-pixel certified non-paired page in each fixed source "
                "category, distinct scores; no prediction inspection."
            ),
            "limitation": (
                "Negative scope controls, " "not untouched model test or corpus accuracy estimate."
            ),
        },
    )
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
