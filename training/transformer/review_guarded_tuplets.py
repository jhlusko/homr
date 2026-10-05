"""Render all nine successful-repair controls with original crops and audited MuseScore panels."""

# ruff: noqa: E501

import argparse
import html
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageChops

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.onnx.score_full_page_musicxml import staff_events
from training.transformer.audit_cached_joint_decoding import sha
from training.transformer.review_cached_joint_decoding import audit_xml, restore

PANELS = {
    "reference": "Reference labels",
    "raw": "Accepted model · before cleanup",
    "standard": "Current cleanup · correct here",
    "guard": "First guard · rejected",
}
NAMES = {
    "6": "quarter-note triplet",
    "12": "eighth-note triplet",
    "24": "sixteenth-note triplet",
    "3": "half-note triplet",
    "4": "quarter note",
    "8": "eighth note",
    "16": "sixteenth note",
    "2": "half note",
}


def meter_text(rows):
    beats = None
    declarations = []
    for row in rows:
        value = row["rhythm"]
        if value.startswith("timeSignatureBeats_"):
            beats = value.split("_", 1)[1]
        elif value.startswith("timeSignature/"):
            declarations.append((beats or "unknown numerator") + "/" + value.split("/", 1)[1])
            beats = None
    return ", ".join(declarations) if declarations else "No meter declaration"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "previous", "run", "musescore", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-rendered", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.reuse_rendered:
        raise FileExistsError("Refusing to replace review")
    controls = json.loads((args.audit / "controls.json").read_text())["controls"]
    wanted = [c for c in controls if c["kind"] == "false_tuplets_fully_repaired"]
    if len(wanted) != 9:
        raise ValueError("Expected all nine successful-repair controls")
    rows = {
        r["id"]: r
        for r in json.loads((args.audit / "changed-crops.json").read_text())
        if r["arm"] == "accepted_oldcore_v7"
    }
    inputs = {
        json.loads(line)["id"]: json.loads(line)
        for line in (args.run / "holdout.jsonl").read_text().splitlines()
    }
    witnesses = {
        r["id"]: r
        for r in json.loads((args.previous / "tuplet-witnesses.json").read_text())["witnesses"]
    }
    if args.reuse_rendered:
        receipt = json.loads((args.out / "provenance.json").read_text())
        for filename, digest in receipt["inputs"].items():
            if sha(filename) != digest:
                raise ValueError("Cached review inputs changed")
        if sha(args.musescore) != receipt["musescore_sha256"]:
            raise ValueError("Cached renderer changed")
    args.out.mkdir(parents=True, exist_ok=True)
    assets = args.out / "assets"
    assets.mkdir(exist_ok=True)
    os.environ["HOMR_STRUCTURED_EXPORT"] = "beam,stem,tie,slur"
    jobs, cases = [], []
    for i, control in enumerate(wanted, 1):
        row = rows[control["id"]]
        case = dict(
            control, key=f"case-{i:02d}", panels={}, reference_meter=meter_text(row["reference"])
        )
        source = Path(inputs[row["id"]]["image"])
        if sha(source) != inputs[row["id"]]["image_sha256"]:
            raise ValueError("Original crop changed: " + str(source))
        copied = assets / (case["key"] + "-original" + source.suffix)
        shutil.copyfile(source, copied)
        case["original"] = "assets/" + copied.name
        case["image_sha256"] = sha(copied)
        case["changes"] = witnesses[row["id"]]["altered_measures"]
        for name in PANELS:
            values = row["reference"] if name == "reference" else row["variants"][name]
            root = generate_xml(XmlGeneratorArguments(), [restore(values)], "")
            prefix = assets / (case["key"] + "-" + name)
            xml = prefix.with_suffix(".musicxml")
            serialized = ET.tostring(root, encoding="unicode")
            if args.reuse_rendered and xml.read_text() != serialized:
                raise ValueError("Cached reconstruction changed: " + str(xml))
            xml.write_text(serialized)
            exported = prefix.with_suffix(".exported.musicxml")
            png = prefix.with_suffix(".png")
            jobs.extend([{"in": str(xml), "out": str(png)}, {"in": str(xml), "out": str(exported)}])
            times = [
                t.findtext("beats", "?") + "/" + t.findtext("beat-type", "?")
                for t in root.findall(".//attributes/time")
            ]
            case["panels"][name] = {
                "xml": str(xml),
                "png": str(png),
                "exported": str(exported),
                "display_meter": ", ".join(times),
                "model_meter": meter_text(values),
            }
        cases.append(case)
        print("Prepared", case["key"], Path(row["id"]).name, flush=True)
    jobfile = args.out / "render-jobs.json"
    jobfile.write_text(json.dumps(jobs, indent=2))
    command = [
        "xvfb-run",
        "-a",
        str(args.musescore),
        "--appimage-extract-and-run",
        "--force",
        "-j",
        str(jobfile),
    ]
    if not args.reuse_rendered:
        with (args.out / "musescore.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900, check=True)
    cards = []
    for case in cases:
        panels = []
        for name, panel in case["panels"].items():
            png = Path(panel["png"])
            pages = [png] if png.exists() else sorted(assets.glob(png.stem + "-*.png"))
            if not pages or not Path(panel["exported"]).exists():
                raise ValueError("Missing renderer output: " + str(png))
            urls = []
            for page in pages:
                im = Image.open(page).convert("RGBA")
                im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
                box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
                if box:
                    im = im.crop(
                        (
                            max(0, box[0] - 16),
                            max(0, box[1] - 16),
                            min(im.width, box[2] + 16),
                            min(im.height, box[3] + 16),
                        )
                    )
                dest = page.with_name(page.stem + "-trim.jpg")
                im.save(dest, quality=92)
                urls.append("assets/" + dest.name)
            audit = audit_xml(Path(panel["xml"]).read_text(), Path(panel["exported"]).read_text())
            original_events = staff_events(ET.parse(panel["xml"]).getroot())
            exported_events = staff_events(ET.parse(panel["exported"]).getroot())
            audit["visible_notes_and_rests_changed"] = original_events != exported_events
            audit["visible_rests_before_after"] = [
                sum(event["pitch"] is None for stream in events for event in stream)
                for events in (original_events, exported_events)
            ]
            panel.update(images=urls, audit=audit)
            errors = (
                "Reference"
                if name == "reference"
                else str(case["metrics"][name]["rhythm"]) + " rhythm errors"
            )
            warning = (
                "⚠ MuseScore changes note/rest timing or placement. Use the original crop and described rhythms."
                if audit["visible_notes_and_rests_changed"]
                else "✓ Visible notes and rests preserve exported timing and placement"
            )
            pictures = "".join(
                f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a>'
                for url in urls
            )
            panels.append(
                f'<section><h3>{PANELS[name]}</h3><b>{errors}</b><p class="small">Meter in labels/prediction: {html.escape(panel["model_meter"])}<br>Display meter: {html.escape(panel["display_meter"])} (XML reconstruction)</p>{pictures}<p class="small">{warning}</p><a class="small" href="assets/{Path(panel["xml"]).name}">Input MusicXML</a></section>'
            )
        changes = []
        for change in case["changes"]:
            transitions = []
            for transition in change["rhythm_transitions"]:
                before = transition["before"].split("_", 1)[1]
                after = transition["after"].split("_", 1)[1]
                transitions.append(
                    f'{transition["count"]} × {NAMES.get(before,before)} → {NAMES.get(after,after)}'
                )
            changes.append(
                f'<li><b>Crop-relative measure {change["crop_relative_measure"]}</b>: {html.escape("; ".join(transitions))}. Predicted length before cleanup: {change["predicted_duration_whole_notes"]} whole note; old median: {change["typical_duration_whole_notes"]}.</li>'
            )
        name = html.escape(Path(case["id"]).name)
        badge = (
            "Meter emitted: " + meter_text(rows[case["id"]]["variants"]["raw"])
            if case["predicted_meter_tokens"]
            else "No meter emitted in this crop decode"
        )
        cards.append(
            f'<article id="{case["key"]}"><header><span class="badge">Case {len(cards)+1} · OSSQ · accepted stack</span><h2>{name}</h2><p><b>{html.escape(badge)}</b> · Reference labels: {html.escape(case["reference_meter"])}</p><ul>{"".join(changes)}</ul></header><div class="original"><h3>Original scanned staff crop — authoritative visual evidence</h3><a href="{case["original"]}" target="_blank"><img src="{case["original"]}"></a></div><div class="panels">{"".join(panels)}</div></article>'
        )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tuplet repair · nine controls</title><style>
body{margin:0;background:#f3f5f7;color:#182738;font:16px/1.55 system-ui}main{max-width:1600px;margin:auto;padding:30px}h1{font-size:34px;margin-bottom:8px}h2{overflow-wrap:anywhere;font-size:21px}h3{font-size:17px}.intro,article{background:white;border:1px solid #dce3e9;border-radius:12px;padding:24px;margin:24px 0}.badge{background:#e5eefc;color:#234c85;border-radius:20px;padding:5px 12px;font-size:13px}.small{font-size:12px;color:#526375}.original{padding:16px;background:#fafafa;border-radius:8px}.original img{width:100%;height:auto}.panels{display:grid;grid-template-columns:repeat(4,minmax(300px,1fr));gap:18px;overflow-x:auto;padding:20px 0}.panels section{border-top:4px solid #4e7d9f;padding:12px;background:#fafbfc}.panels img{width:100%;height:auto}a{color:#255e92}li{margin:8px 0}.notice{border-left:4px solid #d79735;padding:12px 18px;background:#fff8ea}nav{display:flex;gap:14px;flex-wrap:wrap}@media(max-width:700px){main{padding:12px}.intro,article{padding:16px}}
</style><main><span class="badge">Runtime diagnostics · no model promotion</span><h1>When the old tuplet cleanup really helps</h1><div class="intro"><p><b>“Eight of nine have no predicted meter” means these standalone crop decodes emitted no time-signature declaration.</b> It does not mean the printed score has no meter, that meter is absent earlier in the piece, or that full-page decoding has no inherited context.</p><p>These nine accepted-stack crops are the negative controls: the old cleanup makes each exactly match the reference legacy symbol labels. The first guard preserves correct tuplets elsewhere, but fails these useful repairs. Review the original image first, then compare the raw prediction, current cleanup and rejected guard.</p><p class="notice">MuseScore 4.6.5 reconstructions can display a time signature inferred by the XML exporter even when the model emitted none. <b>A visible time signature in a reconstructed panel is not evidence of a predicted meter.</b> Measure numbers below are relative to this crop, not the full score. Reference labels are published crop labels, not an independent full-page MusicXML certificate.</p><p>This review is for understanding whether the repairs agree with the scans and where context is missing. No individual voting or approval is needed. Click any image for full resolution. Rhythmic advance is disabled equally in all panels; renderer changes are flagged.</p><nav>"""
        + "".join(f'<a href="#case-{i:02d}">Case {i}</a>' for i in range(1, 10))
        + "</nav></div>"
        + "".join(cards)
        + "</main></html>"
    )
    (args.out / "index.html").write_text(document)
    (args.out / "cases.json").write_text(json.dumps(cases, indent=2))
    (args.out / "provenance.json").write_text(
        json.dumps(
            {
                "tool_sha256": sha(__file__),
                "musescore_sha256": sha(args.musescore),
                "inputs": {
                    str(p): sha(p)
                    for p in [
                        args.audit / "controls.json",
                        args.audit / "changed-crops.json",
                        args.previous / "tuplet-witnesses.json",
                        args.run / "holdout.jsonl",
                    ]
                },
                "panels": 36,
                "cases": 9,
                "command": command,
                "rendered_assets_reused": args.reuse_rendered,
            },
            indent=2,
        )
    )
    print("Review complete:", args.out, flush=True)


if __name__ == "__main__":
    main()
