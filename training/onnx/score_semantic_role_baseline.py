"""Evaluate frozen detector/OCR selections against separately inspected scan roles.

The existing gate emits Lyrics candidates and abstains otherwise. It cannot
positively classify DirectionText or OtherText. References are evaluation-only.
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.onnx.build_semantic_expansion_review import bounds, reference_overlap
from training.onnx.qualify_joint_export import sha, write

ROLES = ("Lyrics", "DirectionText", "OtherText")


def assess(reference, selected, rejected):
    """Join reference boxes only after inference; never reinterpret abstention."""
    matches = []
    for chosen, rows in ((True, selected), (False, rejected)):
        for row in rows:
            overlap = reference_overlap(reference["box"], bounds(row["box"]))
            if overlap >= 0.5:
                matches.append(
                    dict(
                        text=row["text"],
                        box=row["box"],
                        selected=chosen,
                        reference_coverage=overlap,
                        detector_coverage=row["lyrics_rectangle_union_coverage"],
                    )
                )
    accepted = any(row["selected"] for row in matches)
    return dict(
        prediction="Lyrics" if accepted else "Unknown",
        stage="lyric-candidate" if accepted else "gate-abstention" if matches else "ocr-miss",
        matched_ocr=matches,
    )


def summarize(records):
    summary = {}
    for role in ROLES:
        rows = [row for row in records if row["reference_role"] == role]
        counts = Counter(row["stage"] for row in rows)
        selected = counts["lyric-candidate"]
        summary[role] = dict(
            support=len(rows),
            lyric_candidates=selected,
            gate_abstentions=counts["gate-abstention"],
            ocr_misses=counts["ocr-miss"],
            lyric_selection_rate=selected / len(rows) if rows else None,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = json.loads(args.pages.read_text())
    if any(page["partition"] != "calibration" for page in pages):
        raise ValueError("This diagnostic is calibration-only")
    identity = {page["score"]: page for page in pages}
    inventory = json.loads(args.inventory.read_text())
    gates, hashes = {}, {}
    for score, page in identity.items():
        if sha(Path(page["image"])) != page["image_sha256"]:
            raise ValueError("Scan identity changed")
        path = args.gate / (score + ".json")
        hashes[str(path)] = sha(path)
        prediction = json.loads(path.read_text())
        if prediction["image_sha256"] != page["image_sha256"]:
            raise ValueError("Prediction page identity differs")
        if prediction["source_text_or_reference_geometry_used"]:
            raise ValueError("Reference-dependent selection is forbidden")
        if prediction["selection"] != {"min_area": 200, "minimum_union_coverage": 0.1}:
            raise ValueError("Use the existing fixed gate, not a tuned variant")
        gates[score] = prediction
    records = []
    for reference in inventory:
        if reference["reviewed_role"] not in ROLES:
            raise ValueError("Only inspected roles may enter evaluation")
        page = identity[reference["score"]]
        if reference["page_id"] != page["id"] or reference["image_sha256"] != page["image_sha256"]:
            raise ValueError("Reference page identity differs")
        for key in ("crop", "context"):
            if sha(args.gallery / reference[key]) != reference[key + "_sha256"]:
                raise ValueError("Inspected crop changed")
        prediction = gates[reference["score"]]
        records.append(
            dict(
                id=reference["id"],
                score=reference["score"],
                reference_role=reference["reviewed_role"],
                reference_subtype=reference["reviewed_subtype"],
                box=reference["box"],
                crop=reference["crop"],
                context=reference["context"],
                **assess(reference, prediction["all_page_text"], prediction["rejected_page_text"]),
            )
        )
    by_score = {score: summarize([r for r in records if r["score"] == score]) for score in identity}
    # Multiple reference regions can share an OCR row. Explicitly report this
    # dependency, including mixed-role rows; never call region counts row precision.
    row_references = defaultdict(list)
    for record in records:
        for match in record["matched_ocr"]:
            key = (record["score"], json.dumps(match["box"]))
            row_references[key].append((record["id"], record["reference_role"], match["selected"]))
    shared_rows = [
        dict(score=k[0], box=json.loads(k[1]), references=v)
        for k, v in row_references.items()
        if len(v) > 1
    ]
    report = dict(
        summary=summarize(records),
        by_score=by_score,
        cases=records,
        matched_distinct_ocr_rows=len(row_references),
        shared_ocr_rows=shared_rows,
        reference_matching_minimum_coverage=0.5,
        inference_gate={"minimum_area": 200, "minimum_ocr_union_coverage": 0.1},
        inputs_sha256={
            str(args.inventory): sha(args.inventory),
            str(args.pages): sha(args.pages),
            **hashes,
        },
        code_sha256=sha(Path(__file__)),
        limitation="Inspected calibration regions, not exhaustive page labels or independent test. "
        "Codex role inspection, not user adjudication. Counts may share OCR rows. "
        "Lyric selection is not note attachment; abstention is not a correct direction prediction. "
        "The existing gate is a baseline, not a trained semantic classifier. "
        "No precision estimate across unannotated page text.",
    )
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "gallery").symlink_to(args.gallery.resolve(), target_is_directory=True)
    write(args.out / "report.json", report)
    escape = html.escape
    table = "".join(
        f"<tr><td>{escape(role)}</td><td>{s['support']}</td><td>{s['lyric_candidates']}</td>"
        f"<td>{s['gate_abstentions']}</td><td>{s['ocr_misses']}</td></tr>"
        for role, s in report["summary"].items()
    )
    cards = []
    for record in records:
        matches = (
            "; ".join(
                f"{m['text']} ({'selected' if m['selected'] else 'abstained'})"
                for m in record["matched_ocr"]
            )
            or "No covering full-page OCR row"
        )
        cards.append(
            f"<article><h2>{escape(record['score'])}: {escape(record['reference_role'])} → "
            f"{escape(record['prediction'])}</h2><p>{escape(record['reference_subtype'])}; "
            f"{escape(record['stage'])}</p><p>{escape(matches)}</p><div class='pair'>"
            f"<img src='gallery/{escape(record['crop'])}'>"
            f"<img src='gallery/{escape(record['context'])}'></div></article>"
        )
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Lyric role baseline</title>"
        "<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px}"
        "article{border:1px solid #ccc;padding:18px;margin:24px 0}.pair{display:grid;"
        "grid-template-columns:1fr 2fr;gap:16px}img{max-width:100%}td,th{padding:10px;text-align:left}"
        "@media(max-width:750px){.pair{grid-template-columns:1fr}}</style>"
        "<h1>Existing lyric selection versus inspected roles</h1>"
        f"<p>{escape(report['limitation'])}</p><p>Fixed pinned-e2 detector, area200 and "
        "10% OCR-region overlap. Reference boxes are joined after inference, at 50% "
        "reference coverage. No thresholds changed. Unknown means abstention.</p>"
        "<table><tr><th>Reference role</th><th>Regions</th><th>Selected as lyrics</th>"
        "<th>Gate abstentions</th><th>OCR misses</th></tr>" + table + "</table>"
        "<p><a href='report.json'>Full results and receipts</a></p>" + "".join(cards)
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
