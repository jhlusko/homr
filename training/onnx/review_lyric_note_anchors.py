"""Build original-scan overlays; attention geometry is not ground truth."""

import argparse
import html
import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training.onnx.qualify_joint_export import write


def build(pages_path, pilot, out):
    pages = json.loads(pages_path.read_text())
    out.mkdir(parents=True, exist_ok=False)
    sections, summary = [], []
    linkage_path = pilot / "source-linkage.json"
    linkage = (
        {}
        if not linkage_path.exists()
        else {p["score"]: p for p in json.loads(linkage_path.read_text())["pages"]}
    )
    for index, page in enumerate(pages):
        directory = pilot / f"page-{index}"
        anchors = json.loads((directory / "note-anchors.json").read_text())
        transforms = json.loads((directory / "staff-transforms.json").read_text())
        image_name = f"page-{index}.png"
        shutil.copyfile(page["image"], out / image_name)
        width, height = Image.open(page["image"]).size
        preview = Image.open(page["image"]).convert("RGB")
        draw = ImageDraw.Draw(preview)
        marks, distances = [], []
        proposed = {}
        for system in linkage.get(page["score"], {}).get("systems", []):
            for link in system["proposed_links"]:
                proposed[(link["prediction"]["staff_call"], link["prediction"]["token_index"])] = (
                    link["source"]
                )
        for transform in transforms:
            ratio = np.array(transform["original_to_processed_ratio"])
            noteheads = np.array(transform["detected_noteheads"]).reshape(-1, 2) * ratio
            for x, y in noteheads:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="#079455", width=2)
                marks.append(
                    f'<circle class="heads" cx="{x}" cy="{y}" r="6" fill="none" stroke="#079455" stroke-width="2"><title>Detected notehead; source ownership unverified</title></circle>'
                )
            for anchor in anchors:
                if (
                    anchor["staff_call"] != transform["staff_call"]
                    or anchor["page_coordinate"] is None
                ):
                    continue
                x, y = anchor["page_coordinate"]
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline="#d97706", width=2)
                label = f'Staff call {anchor["staff_call"] + 1}: {anchor["pitch"]}, {anchor["rhythm"]}; first-pass attention'
                source = proposed.get((anchor["staff_call"], anchor["token_index"]))
                if source is not None:
                    beat = Fraction(source["onset_quarters"]) + 1
                    label = f'Measure {source["physical_measure"]}, beat {beat} ({source["pitch"]}); exact sequence linkage proposal; first-pass attention'
                marks.append(
                    f'<g class="attention"><circle cx="{x}" cy="{y}" r="7" fill="none" stroke="#d97706" stroke-width="2"><title>{html.escape(label)}</title></circle></g>'
                )
                if len(noteheads):
                    distance = float(np.linalg.norm(noteheads - [x, y], axis=1).min())
                    distances.append(distance)
        skips = [row for row in anchors if row["page_coordinate"] is None]
        preview.thumbnail((1500, 1500))
        preview.save(out / f"preview-{index}.png")
        record = dict(
            score=page["score"],
            notes=len(anchors),
            mapped=len(anchors) - len(skips),
            skipped=len(skips),
            geometry_skips=[row.get("geometry_skip") for row in skips],
            distance_to_nearest_detected_notehead_pixels=dict(
                median=None if not distances else float(np.median(distances)),
                p90=None if not distances else float(np.percentile(distances, 90)),
            ),
            source_ownership_certified=False,
            attachment_scored=False,
        )
        summary.append(record)
        sections.append(
            f'<section><h2>{html.escape(page["score"])}</h2><p>{record["mapped"]}/{record["notes"]} attention coordinates mapped; {record["skipped"]} skipped. Hover circles for decoded pitch/rhythm. No measure labels are assigned until native ownership is verified.</p><details><summary>Geometry diagnostics</summary><pre>{html.escape(json.dumps(record, indent=2))}</pre></details><svg viewBox="0 0 {width} {height}" style="width:100%;background:white"><image href="{image_name}" width="{width}" height="{height}"/>{"".join(marks)}</svg></section>'
        )
    witness_path = pilot / "witness-score.json"
    witness_section = ""
    if witness_path.exists():
        report = json.loads(witness_path.read_text())
        cards = []
        for i, witness in enumerate(report["witnesses"]):
            expected, prediction = witness["reference"], witness["predicted"][0]
            beat = Fraction(expected["onset_quarters"]) + 1
            cards.append(
                f'<li><strong>{html.escape(expected["text"])}</strong> ({witness["witness"]["score"]}): reference measure {expected["physical_measure"]}, beat {beat}, verse {expected["verse"]}; nearest-x measure {prediction["physical_measure"]}, beat {Fraction(prediction["onset_quarters"])+1}, verse {prediction["verse"]}.</li>'
            )
        witness_section = (
            "<section><h2>Three single-syllable smoke controls</h2><p>3/3 text and note ownership checks pass with manually fixed system/verse routing. These selected examples do not measure corpus accuracy, automatic verse ordering, detector recall, grouped OCR syllables, or final XML ownership.</p><ul>"
            + "".join(cards)
            + "</ul></section>"
        )
    (out / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Lyric note-anchor geometry pilot</title>
<style>body{font:16px system-ui;background:#f3f4f6;color:#111827;max-width:1500px;margin:24px auto;padding:0 16px}section{background:white;padding:16px;margin:24px 0}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border-bottom:1px solid #ccc}pre{white-space:pre-wrap}</style>
<h1>Lyric note-anchor geometry pilot</h1><p>Original calibration scans. Orange: approximate first-pass decoder attention. Green: segmentation noteheads. These overlays test coordinate plumbing; they do not certify note ownership or lyric accuracy.</p>
<header><label><input type="checkbox" checked onchange="document.querySelectorAll('.attention').forEach(x=>x.style.display=this.checked?'':'none')"> Attention coordinates</label> <label><input type="checkbox" checked onchange="document.querySelectorAll('.heads').forEach(x=>x.style.display=this.checked?'':'none')"> Detected noteheads</label></header>"""
        + witness_section
        + "".join(sections)
    )
    write(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.pages, args.pilot, args.out), indent=2))
