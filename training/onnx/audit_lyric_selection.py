"""Audit cached selection against fixed calibration rows; never changes selection."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image

from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_lyric_rows import row_for_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("references", "pages", "ocr", "gate", "out", "review"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.review.exists():
        raise ValueError("Outputs must be new")
    refs = json.loads(args.references.read_text())
    receipt = json.loads((args.references.parent / "receipt.json").read_text())
    if receipt["partition"] != "calibration" or receipt["rows_sha256"] != sha(args.references):
        raise ValueError("Reference identity mismatch")
    pages = {p["score"]: p for p in json.loads(args.pages.read_text())}
    args.out.mkdir(parents=True)
    args.review.mkdir(parents=True)
    summaries, cases, inputs = [], [], {}
    for variant in sorted(args.gate.glob("area*-coverage*")):
        for score, page in pages.items():
            path = variant / (score + ".json")
            selected = json.loads(path.read_text())
            raw_path = args.ocr / (score + ".json")
            raw = json.loads(raw_path.read_text())
            if (sha(raw_path) != selected["original_ocr_sha256"]
                    or sha(page["image"]) != selected["image_sha256"]
                    or raw["image_sha256"] != selected["image_sha256"]):
                raise ValueError("Input identities differ")
            rows = [r for r in refs if r["score"] == score]
            if any(r["image_sha256"] != selected["image_sha256"] for r in rows):
                raise ValueError("Reference page differs")
            inputs[str(path)] = sha(path)
            counts = {"retained_inside_rows": 0, "retained_outside_rows": 0,
                      "dropped_inside_rows": 0, "dropped_outside_rows": 0}
            for status, records in (("retained", selected["all_page_text"]),
                                    ("dropped", selected["rejected_page_text"])):
                for item in records:
                    row = row_for_box(item["box"], rows)
                    category = status + ("_inside_rows" if row else "_outside_rows")
                    counts[category] += 1
                    if variant.name == "area200-coverage0.1" and category in (
                            "retained_outside_rows", "dropped_inside_rows"):
                        cases.append(dict(item, score=score, category=category, row_id=row))
            if sum(counts.values()) != len(raw["all_page_text"]):
                raise ValueError("Selection lost regions")
            summaries.append(dict(variant=variant.name, score=score, **counts))
    parts = ['<!doctype html><meta charset="utf-8"><title>Lyric selection audit</title>',
             '<style>body{font:17px system-ui;max-width:1100px;margin:30px auto}',
             'article{border:1px solid #bbb;padding:18px;margin:20px 0}',
             'img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}</style>',
             '<h1>Lyric selection: retained non-row regions and dropped lyric fragments</h1>',
             '<p>Predeclared area-200 / 10% bounding-rectangle coverage, two calibration pages. ',
             'Selection used no reference text or rows. Manual rows are used only for this audit. ',
             'Outside-row regions require visual adjudication; they are not automatically false positives. ',
             'No note attachment or release decision is requested.</p>',
             '<table><tr><th>Page</th><th>Retained in lyric rows</th><th>Retained outside</th>',
             '<th>Dropped in lyric rows</th><th>Dropped outside</th></tr>']
    for row in summaries:
        if row["variant"] == "area200-coverage0.1":
            parts.append('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in (
                "score", "retained_inside_rows", "retained_outside_rows",
                "dropped_inside_rows", "dropped_outside_rows")) + '</tr>')
    parts.append('</table>')
    for index, case in enumerate(cases):
        image_path = pages[case["score"]]["image"]
        with Image.open(image_path) as image:
            xs, ys = [p[0] for p in case["box"]], [p[1] for p in case["box"]]
            box = (max(0, int(min(xs)) - 30), max(0, int(min(ys)) - 30),
                   min(image.width, int(max(xs)) + 31), min(image.height, int(max(ys)) + 31))
            crop_path = args.review / f'case-{index:03}.png'
            image.crop(box).save(crop_path)
        case.update(crop=crop_path.name, crop_sha256=sha(crop_path), context_box=box)
        label = "Dropped fragment in certified lyric row" if case["row_id"] else "Retained region outside certified lyric rows"
        parts.append(f'<article><h2>{html.escape(case["score"])} — {label}</h2>'
                     f'<p>OCR: {html.escape(case["text"])}; rectangle coverage '
                     f'{case["lyrics_rectangle_union_coverage"]:.1%}; '
                     f'row: {html.escape(case["row_id"] or "not assigned")}</p>'
                     f'<img src="{crop_path.name}"></article>')
    (args.review / "index.html").write_text('\n'.join(parts))
    write(args.out / "report.json", {"summary": summaries, "cases": cases,
          "reference_sha256": sha(args.references), "inputs_sha256": inputs,
          "source_sha256": sha(Path(__file__)), "scope": "Fixed-row spatial calibration audit; "
          "not semantic precision or note attachment. Representative setting predeclared."})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
