"""Original-scan comparisons for recovered OCR rows and changed staff positions."""

import argparse
import html
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.lyric_row_routing import physical_staffs
from training.onnx.qualify_joint_export import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--two-page", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cards = []

    def save_case(image_path, transforms, bounds, draw_details, title, explanation):
        original = Image.open(image_path).convert("RGB")
        left, top, right, bottom = bounds
        bounds = (
            max(0, int(left)),
            max(0, int(top)),
            min(original.width, int(right)),
            min(original.height, int(bottom)),
        )
        annotated = original.copy()
        draw = ImageDraw.Draw(annotated)
        for staff in physical_staffs(transforms):
            for line_index in range(5):
                draw.line([(x, ys[line_index]) for x, ys in staff["grid"]], fill="#168aad", width=2)
        draw_details(draw)
        index = len(cards)
        original.crop(bounds).save(args.out / f"case-{index}-original.png")
        annotated.crop(bounds).save(args.out / f"case-{index}-overlay.png")
        cards.append(
            f'<article><h2>{html.escape(title)}</h2><p>{html.escape(explanation)}</p><div class="pair"><figure><figcaption>Original scan</figcaption><img src="case-{index}-original.png"></figure><figure><figcaption>Diagnostic geometry</figcaption><img src="case-{index}-overlay.png"></figure></div></article>'
        )

    completed = json.loads((args.expanded / "completion.json").read_text())
    for page in completed["pages"]:
        before = json.loads((args.expanded / f"page-{page['page_index']}/routing.json").read_text())
        transforms = json.loads(
            (args.expanded / f"page-{page['page_index']}/staff-transforms.json").read_text()
        )
        after = json.loads((args.followup / f"{page['score']}-routing.json").read_text())
        for recovered in after["routed"]:
            if "recovery" not in recovered:
                continue
            row = after["rows"][recovered["index"]]
            xs, ys = zip(*row["box"])

            def overlay(draw, row=row, recovered=recovered, rows=after["rows"]):
                draw.polygon([tuple(p) for p in row["box"]], outline="#e76f00", width=4)
                for index in recovered["seed_indices"]:
                    draw.polygon([tuple(p) for p in rows[index]["box"]], outline="#009944", width=3)

            explanation = "Orange: recovered OCR region. Green: strict-pass neighbors on the same physical row. Blue: segmented staff lines. Entire orange box stays between staff lines; only the extra margin is relaxed. Native verse and lyric role remain unqualified."
            if row["text"] == "tempo.":
                explanation += (
                    " This is the a tempo direction control: routing it does not make it a lyric."
                )
            save_case(
                before["page"]["image"],
                transforms,
                (min(xs) - 120, min(ys) - 170, max(xs) + 220, max(ys) + 100),
                overlay,
                f"{page['score']} — {row['text']}",
                explanation,
            )
    report = json.loads((args.followup / "geometry-report.json").read_text())
    for note in report["vocal_pitch_audit"]:
        if note["stored_agrees"]:
            continue
        page_index = 0 if note["score"] == "IMSLP154060" else 1
        folder = args.two_page / f"page-{page_index}"
        identity = json.loads((folder / "identity.json").read_text())
        transforms = json.loads((folder / "staff-transforms.json").read_text())
        transform = next(t for t in transforms if t["staff_call"] == note["system"])
        x, y = (
            note["center"][0] * transform["original_to_processed_ratio"][0],
            note["center"][1] * transform["original_to_processed_ratio"][1],
        )

        def overlay(draw, x=x, y=y):
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), outline="#e76f00", width=4)

        native = note["native_reference"]
        source = Path(__file__).resolve().parents[3] / identity["page"]["source"]["source"]
        if sha(source) != identity["page"]["source"]["source_sha256"]:
            raise ValueError("Review native source changed")
        part = next(
            p for p in read_mxl(source).findall("part") if p.get("id") == native["source_part"]
        )
        denominator = None
        for measure in part.findall("measure")[: native["physical_measure"]]:
            time = measure.find("attributes/time")
            if time is not None:
                denominator = int(time.findtext("beat-type"))
        if denominator is None:
            raise ValueError("Review beat label requires native meter")
        beat = 1 + Fraction(native["onset_quarters"]) * denominator / 4
        whole, remainder = divmod(beat.numerator, beat.denominator)
        beat_label = str(whole) if not remainder else f"{whole}+{remainder}/{beat.denominator}"
        save_case(
            identity["page"]["image"],
            transforms,
            (x - 150, y - 140, x + 150, y + 140),
            overlay,
            f"{note['score']} — measure {native['physical_measure']}, beat {beat_label} ({native['pitch']})",
            f"Stored detector staff position: {note['stored_staff_position']}; interpolated: {note['staff_position']} (continuous {note['continuous_position']:.3f}); decoded/reference: {note['decoded_position']}. Positions count half-line steps from the bottom staff line (1). This changes diagnostic pitch evidence, not recognition or the exported score.",
        )
    document = (
        '<!doctype html><meta charset="utf-8"><title>Lyric geometry follow-up</title><style>body{font:16px system-ui;margin:24px;background:#f6f7f8;color:#17212b}article{background:white;padding:20px;margin:24px 0;border-radius:8px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{max-width:100%;height:auto}figcaption{margin-bottom:8px;font-weight:bold}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><h1>Lyric geometry follow-up</h1><p>Ten recovered physical rows (including the a tempo direction control) and three changed pitch-geometry cases. Original scans appear beside overlays. These are calibration diagnostics; native verse assignment, text-role classification and production attachment remain unqualified. No individual approval decisions are requested.</p>'
        + "".join(cards)
    )
    (args.out / "index.html").write_text(document)
    print(f"Wrote {len(cards)} comparisons")


if __name__ == "__main__":
    main()
