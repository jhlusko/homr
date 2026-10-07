"""Review frozen semantic positives and unknown full-page OCR with scan context."""

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from training.onnx.qualify_joint_export import sha, write


def bounds(polygon):
    return [
        min(p[0] for p in polygon),
        min(p[1] for p in polygon),
        max(p[0] for p in polygon),
        max(p[1] for p in polygon),
    ]


def reference_overlap(reference, observed):
    x, y, xx, yy = reference
    a, b, aa, bb = observed
    area = max(0, min(xx, aa) - max(x, a)) * max(0, min(yy, bb) - max(y, b))
    return area / max(1, (xx - x) * (yy - y))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    args = parser.parse_args()
    decisions = (
        {r["id"]: r for r in json.loads(args.adjudication.read_text())["decisions"]}
        if args.adjudication
        else {}
    )
    pages = json.loads((args.inputs / "evaluation-pages.json").read_text())
    ocr_receipt = json.loads((args.inputs / "full-ocr/receipt.json").read_text())
    if ocr_receipt["pages_sha256"] != sha(args.inputs / "inference-pages.json"):
        raise ValueError("OCR input identity changed")
    args.out.mkdir(parents=True, exist_ok=False)
    cases, cards, ocr_hashes = [], [], {}
    for page in pages:
        image_path = Path(page["image"])
        if sha(image_path) != page["image_sha256"]:
            raise ValueError("Original scan changed")
        ocr_path = args.inputs / "full-ocr" / (page["score"] + ".json")
        ocr_hashes[str(ocr_path)] = sha(ocr_path)
        ocr = json.loads(ocr_path.read_text())
        if ocr["image_sha256"] != page["image_sha256"] or ocr["source_text_passed_to_ocr"]:
            raise ValueError("OCR provenance differs")
        image = Image.open(image_path).convert("RGB")
        positives = []
        for family in ("DirectionText", "Lyrics"):
            family_rows = [r for r in page["ground_truth"] if r["semanticFamily"] == family]
            family_rows.sort(
                key=lambda r: hashlib.sha256(json.dumps(r["box"]).encode()).hexdigest()
            )
            positives.extend(family_rows if family == "DirectionText" else family_rows[:3])
        selected = [
            dict(family=r["semanticFamily"], box=r["box"], evidence=r["evidence"])
            for r in positives
        ]
        unknown = []
        for row in ocr["all_page_text"]:
            box = bounds(row["box"])
            if (
                len([c for c in row["text"] if c.isalpha()]) < 3
                or row["confidence"] < 0.6
                or box[1] < page["height"] * 0.15
            ):
                continue
            if any(reference_overlap(r["box"], box) >= 0.5 for r in page["ground_truth"]):
                continue
            unknown.append(dict(family="Unknown", box=box, evidence=[], ocr_row=row))
        unknown.sort(key=lambda r: hashlib.sha256(json.dumps(r["box"]).encode()).hexdigest())
        selected.extend(unknown[:2])
        for index, record in enumerate(selected):
            box = list(map(int, record["box"]))
            x, y, xx, yy = box
            stem = page["score"] + "-" + str(index)
            tight = [
                max(0, x - 20),
                max(0, y - 15),
                min(image.width, xx + 20),
                min(image.height, yy + 15),
            ]
            context = [0, max(0, y - 230), image.width, min(image.height, yy + 230)]
            crop_path, context_path = args.out / (stem + "-text.png"), args.out / (
                stem + "-context.jpg"
            )
            image.crop(tight).save(crop_path)
            band = image.crop(context)
            ImageDraw.Draw(band).rectangle(
                [x, y - context[1], xx, yy - context[1]], outline="#1473ba", width=4
            )
            band.save(context_path, quality=92)
            matching = [
                r for r in ocr["all_page_text"] if reference_overlap(box, bounds(r["box"])) >= 0.5
            ]
            reference_text = [e["text"] for e in record["evidence"] if e.get("text")]
            case = dict(
                id=stem,
                score=page["score"],
                page_id=page["id"],
                image=str(image_path),
                image_sha256=sha(image_path),
                box=box,
                candidate_role=record["family"],
                reference_evidence=record["evidence"],
                reference_text=reference_text,
                ocr_text=[r["text"] for r in matching],
                crop=crop_path.name,
                context=context_path.name,
                crop_sha256=sha(crop_path),
                context_sha256=sha(context_path),
                role_adjudication="pending original-scan inspection",
                negative_label=False,
            )
            cases.append(case)
            if decisions:
                decision = decisions[stem]
                for key in ("crop_sha256", "context_sha256", "image_sha256", "box"):
                    if decision[key] != case[key]:
                        raise ValueError("Adjudicated scan region changed")
                case.update(
                    reviewed_role=decision["role"],
                    reviewed_subtype=decision["subtype"],
                    role_adjudication="Codex original-scan inspection; not user review",
                )
            role = {
                "DirectionText": "Direction candidate",
                "Lyrics": "Lyric candidate",
                "Unknown": "Unannotated — role unknown",
            }[record["family"]]
            displayed_role = case.get("reviewed_role", record["family"])
            if decisions:
                role = {
                    "DirectionText": "Direction",
                    "Lyrics": "Sung lyric",
                    "OtherText": "Dedication / credit",
                }[displayed_role]
                role += " — Codex scan inspection"
            cards.append(
                f"<article data-role='{displayed_role}'><h2>{html.escape(page['score'])} — {role}</h2>"
                f"<p>OCR: <strong>{html.escape(' | '.join(case['ocr_text']) or '(no matching line)')}</strong></p>"
                f"<p class='muted'>Existing positive text evidence: {html.escape(' | '.join(reference_text) or 'none; role annotation only or unannotated')}</p>"
                f"<div class='pair'><figure><figcaption>Original text crop</figcaption><img src='{crop_path.name}'></figure>"
                f"<figure><figcaption>Original staff context — outlined region</figcaption><a href='{context_path.name}'><img src='{context_path.name}'></a></figure></div>"
                f"<p><a href='{context_path.name}' target='_blank'>Open full-resolution context</a> · Region {html.escape(str(box))}</p></article>"
            )
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Broader lyric and direction controls</title>
<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;background:#f3f6f9;color:#1d2938}p{line-height:1.6}article{background:white;padding:22px;border:1px solid #d3dce5;border-radius:10px;margin:24px 0}.pair{display:grid;grid-template-columns:1fr 2fr;gap:20px}img{max-width:100%;height:auto}figure{margin:0}figcaption{font-weight:600;margin-bottom:12px}.muted{color:#536579}.notice{background:#fff3d5;padding:18px;border-left:4px solid #c39833}button{font:inherit;padding:8px 12px;margin:5px}a{color:#1767a8}@media(max-width:750px){.pair{grid-template-columns:1fr}}</style>
<h1>Broader lyric / direction controls</h1><p>Eight real Lieder pages outside the previous C1 calibration scores. These include lyric positives, direction positives, and unannotated OCR regions. Full-page OCR received scans only, never references.</p>
<p class="notice"><strong>This is a reference review, not classifier performance.</strong> Existing annotations are positive-only and may need correction. Unannotated does not mean non-lyric. Detector/core training membership is unverified. No formal test pages or classifier threshold tuning.</p>
<p>Check whether the outlined text is sung lyrics, a performance direction, or uncertain. Staff context matters, especially directions between vocal and piano staves. Crops are original scan pixels; the dashboard outline is only a viewing aid.</p>
<div><button onclick="filter('all')">All</button><button onclick="filter('Lyrics')">Lyrics</button><button onclick="filter('DirectionText')">Directions</button><button onclick="filter('OtherText')">Credits / other</button><button onclick="filter('Unknown')">Unreviewed</button></div>
"""
        + "\n".join(cards)
        + """<script>function filter(role){for(const card of document.querySelectorAll('article'))card.hidden=role!=='all'&&card.dataset.role!==role;}</script></html>"""
    )
    (args.out / "index.html").write_text(page)
    if decisions:
        reviewed_counts = Counter(r.get("reviewed_role", "Unknown") for r in cases)
        page = page.replace(
            "<h1>Broader lyric / direction controls</h1>",
            '<h1>Broader lyric / direction controls</h1><p class="notice">'
            f'<strong>Original-pixel inspection by Codex: {reviewed_counts["Lyrics"]} lyric regions, '
            f'{reviewed_counts["DirectionText"]} directions, {reviewed_counts["OtherText"]} other text regions.</strong> '
            "Original positive/candidate evidence is retained. This is not user adjudication, "
            "independent test performance or note-attachment truth.</p>",
        )
        (args.out / "index.html").write_text(page)
    write(args.out / "inventory.json", cases)
    write(
        args.out / "receipt.json",
        dict(
            cases=len(cases),
            candidate_roles=Counter(r["candidate_role"] for r in cases),
            evaluation_sha256=sha(args.inputs / "evaluation-pages.json"),
            ocr_receipt_sha256=sha(args.inputs / "full-ocr/receipt.json"),
            ocr_records_sha256=ocr_hashes,
            code_sha256=sha(Path(__file__)),
            adjudication_sha256=sha(args.adjudication) if args.adjudication else None,
            reviewed_roles=Counter(r.get("reviewed_role", "unreviewed") for r in cases),
            limitation=(
                "Codex scan inspection of semantic roles only; not user adjudication or note-attachment truth. "
                if decisions
                else "Candidate reference roles only; pending scan inspection. "
            )
            + "No classifier performance claim or automatic negative labels.",
        ),
    )


if __name__ == "__main__":
    main()
