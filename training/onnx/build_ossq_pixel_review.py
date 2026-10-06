"""Build scan/source context for every recovered OSSQ pixel difference."""

import argparse
import copy
import hashlib
import html
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image

from training.onnx.prepare_page_certification_extension import preview_window


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_window(root, numbers):
    """Resolve a unique contiguous source window, without assuming number=index."""
    wanted = [str(number) for number in numbers]
    windows = []
    for part in root.findall("part"):
        actual = [m.get("number") for m in part.findall("measure")]
        starts = [
            i for i in range(len(actual) - len(wanted) + 1) if actual[i : i + len(wanted)] == wanted
        ]
        if len(starts) != 1 or not wanted:
            raise ValueError("Source measure labels are not unique and contiguous")
        windows.append((starts[0], starts[0] + len(wanted)))
    if not windows or len(set(windows)) != 1:
        raise ValueError("Source parts disagree on physical measure window")
    return windows[0]


def measure_signature(measure):
    """Visible musical anchors including accidentals, without renderer metadata."""
    return tuple(
        (
            note.findtext("type", ""),
            note.findtext("pitch/step", ""),
            note.findtext("pitch/octave", ""),
            float(note.findtext("pitch/alter", "0")),
            len(note.findall("dot")),
            note.find("chord") is not None,
            note.find("rest") is not None,
            note.find("grace") is not None,
        )
        for note in measure.findall("note")
        if note.get("print-object") != "no"
    )


def resolve_symbolic_window(root, segment):
    """Require one shared physical window matching every source part's music."""
    originals, cuts = root.findall("part"), segment.findall("part")
    if not originals or len(originals) != len(cuts):
        raise ValueError("Source and segment part counts differ")
    starts = None
    lengths = set()
    for original, cut in zip(originals, cuts):
        actual = [measure_signature(m) for m in original.findall("measure")]
        wanted = [measure_signature(m) for m in cut.findall("measure")]
        lengths.add(len(wanted))
        matches = {
            i for i in range(len(actual) - len(wanted) + 1) if actual[i : i + len(wanted)] == wanted
        }
        starts = matches if starts is None else starts & matches
    if len(lengths) != 1 or lengths == {0} or len(starts) != 1:
        raise ValueError("No unique all-part musical source window")
    start = next(iter(starts))
    return start, start + next(iter(lengths))


def resolve_with_neighbors(root, symbolic):
    """Disambiguate repeated passages using exact neighbouring symbolic systems."""
    paths = sorted(symbolic.parent.glob("*.musicxml"))
    index = paths.index(symbolic)
    for radius in range(1, 5):
        selected = paths[max(0, index - radius) : index + radius + 1]
        merged = ET.Element("score-partwise")
        originals = root.findall("part")
        for _ in originals:
            ET.SubElement(merged, "part")
        offset = 0
        length = 0
        for path in selected:
            cut = ET.parse(path).getroot().findall("part")
            counts = {len(part.findall("measure")) for part in cut}
            if len(cut) != len(originals) or len(counts) != 1:
                raise ValueError("Neighbour system parts disagree")
            count = next(iter(counts))
            if path < symbolic:
                offset += count
            if path == symbolic:
                length = count
            for target, source in zip(merged.findall("part"), cut):
                target.extend(copy.deepcopy(source.findall("measure")))
        try:
            start, _ = resolve_symbolic_window(root, merged)
            return start + offset, start + offset + length, selected
        except ValueError:
            continue
    raise ValueError("Source window unresolved even with neighbouring systems")


def annotate_source_discrepancies(out, diagnostics):
    """Add explicit source warnings without treating either source as adjudicated."""
    data = json.loads(diagnostics.read_text())
    rows = "".join(
        "<li>"
        + html.escape(case["system"])
        + " — "
        + html.escape(case["difference"])
        + "; parts "
        + ", ".join(str(n) for n in case["affected_parts"])
        + ".</li>"
        for case in data["cases"]
    )
    warning = (
        "<aside style='background:#fff3d0;padding:16px;border:1px solid #d2a34b'>"
        "<h2>Three unresolved source comparisons</h2><p>These archived segments disagree "
        "with original MusicXML on accidentals. This is separate from the crop-pixel changes; "
        "the correct values still need scan adjudication.</p><ul>" + rows + "</ul></aside>"
    )
    path = out / "index.html"
    page = path.read_text()
    if "Three unresolved source comparisons" in page:
        raise ValueError("Source warnings already attached")
    path.write_text(page.replace("</h1>", "</h1>" + warning, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--musescore", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.out / "assets"
    assets.mkdir()
    validation = json.loads(
        (args.recovery / "validation-pixel-differences-hash-verified.json").read_text()
    )
    borodin = json.loads((args.recovery / "borodin-all-pixel-receipt.json").read_text())
    candidates = json.loads((args.recovery / "crop-mapping-candidates.json").read_text())
    candidate_index = {Path(row["label"]).stem: row for row in candidates["crops"]}
    borodin_map = {
        row["scan_system"]: row
        for row in json.loads(
            (args.recovery / "borodin-candidate-system-mapping.json").read_text()
        )["pairs"]
    }
    rows = [{**row, "corpus_split": "OSSQ validation"} for row in validation["cases"]]
    rows += [
        {**row, "corpus_split": "Borodin training"} for row in borodin["cases"] if not row["equal"]
    ]
    groups = defaultdict(list)
    for row in rows:
        stem = Path(row["public"]).stem.split("-")[0]
        score, page, system, part = stem.split("_")
        key = f"{score}:{page}:{system}"
        row.update(id=stem, score=score, page=int(page), system=int(system), part=int(part))
        groups[key].append(row)
    sources = {
        p.stem: p
        for p in args.archive.glob("scores/*/*/sq*.musicxml")
        if p.stem.startswith("sq") and p.stem[2:].isdigit()
    }
    prepared = []
    for key, cases in sorted(groups.items()):
        score = cases[0]["score"]
        source = sources[score]
        folder = source.parent
        root = ET.parse(source).getroot()
        part_names = [p.findtext("part-name", "Part") for p in root.findall("part-list/score-part")]
        context = dict(
            id=key.replace(":", "_"),
            score=score,
            title=" / ".join(source.parent.parts[-2:]).replace("_", " "),
            source=str(source),
            source_sha256=sha(source),
            cases=cases,
            part_names=part_names,
            certified_scan_ownership=False,
        )
        exemplar = cases[0]
        if score == "sq10675759":
            metadata = Path(borodin_map[key]["metadata"])
            symbolic = (
                folder / "musicxml/unaligned" / (borodin_map[key]["symbolic_segment"] + ".musicxml")
            )
        else:
            metadata = Path(candidate_index[exemplar["id"]]["metadata"])
            symbolic = folder / "musicxml/scanned/systemwise" / (key + ".musicxml")
        data = yaml.safe_load(metadata.read_text())
        context.update(
            measure_labels=data["measure_numbers"],
            metadata=str(metadata),
            metadata_sha256=sha(metadata),
        )
        scan = folder / "images/scanned/original" / f"{score}:{exemplar['page']:04d}.png"
        scan_target = assets / f"{score}_{exemplar['page']:04d}-scan.png"
        if not scan_target.exists():
            shutil.copyfile(scan, scan_target)
        context["scan"] = str(scan_target.relative_to(args.out))
        system_image = folder / "images/scanned/systemwise" / (key + ".png")
        if system_image.is_file():
            target = assets / (context["id"] + "-system.png")
            shutil.copyfile(system_image, target)
            context["system_image"] = str(target.relative_to(args.out))
        for row in cases:
            for side in ("backup", "public"):
                path = Path(row[side])
                expected = row.get(side + "_sha256")
                if expected and sha(path) != expected:
                    raise ValueError(f"Evidence changed: {path}")
                target = assets / f"{row['id']}-{side}.png"
                shutil.copyfile(path, target)
                row[side + "_asset"] = str(target.relative_to(args.out))
                row[side + "_sha256"] = sha(path)
                with Image.open(path) as image:
                    row[side + "_dimensions"] = list(image.size)
        try:
            try:
                start, end = resolve_symbolic_window(root, ET.parse(symbolic).getroot())
            except ValueError:
                start, end, neighbours = resolve_with_neighbors(root, symbolic)
                context["neighbour_evidence"] = {str(p): sha(p) for p in neighbours}
            context["symbolic_segment"] = str(symbolic)
            context["symbolic_sha256"] = sha(symbolic)
            context["source_window_method"] = "unique musical content match across all source parts"
            context["physical_source_window"] = [start, end]
            preview = preview_window(root, start, end)
            xml = assets / (context["id"] + "-source.musicxml")
            ET.ElementTree(preview).write(xml, encoding="utf-8", xml_declaration=True)
            context["source_preview"] = str(xml.relative_to(args.out))
        except ValueError as error:
            context["source_window_error"] = str(error)
        prepared.append(context)
    runtime = args.out / "qt-runtime"
    runtime.mkdir(mode=0o700)
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="offscreen", XDG_RUNTIME_DIR=str(runtime))
    fallback = args.musescore.parent.parent / "fallback"
    libraries = sorted({str(p.parent) for p in fallback.rglob("*.so*") if p.is_file()})
    env["LD_LIBRARY_PATH"] = ":".join(libraries + [env.get("LD_LIBRARY_PATH", "")])

    def render(context):
        if "source_preview" not in context:
            return
        xml = args.out / context["source_preview"]
        pdf = xml.with_suffix(".pdf")
        log = xml.with_suffix(".log")
        try:
            with log.open("w") as stream:
                subprocess.run(
                    [str(args.musescore), "-o", str(pdf), str(xml)],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=True,
                )
                prefix = xml.with_suffix("")
                subprocess.run(
                    ["pdftoppm", "-scale-to", "1400", "-png", str(pdf), str(prefix)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=True,
                )
            images = sorted(xml.parent.glob(xml.stem + "-*.png"))
            context["source_render_pages"] = [str(p.relative_to(args.out)) for p in images]
            context["source_pdf"] = str(pdf.relative_to(args.out))
        except (subprocess.SubprocessError, OSError) as error:
            context["render_error"] = str(error)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(render, prepared))
    cards = []
    for context in prepared:
        measures = ", ".join(str(n) for n in context["measure_labels"])
        comparisons = []
        for row in context["cases"]:
            name = context["part_names"][row["part"] - 1]
            comparisons.append(
                f"<section><h3>{html.escape(name)} · staff {row['part']}</h3>"
                "<div class='pair'>"
                + "".join(
                    f"<figure><figcaption>{label} · {row[side + '_dimensions'][0]} × "
                    f"{row[side + '_dimensions'][1]} pixels</figcaption>"
                    f"<a href='{row[side + '_asset']}' target='_blank'><img loading='lazy' "
                    f"src='{row[side + '_asset']}' alt='{label} crop'></a></figure>"
                    for side, label in (("backup", "August crop"), ("public", "Published v2 crop"))
                )
                + "</div></section>"
            )
        render_html = "".join(
            f"<img loading='lazy' src='{p}' alt='MuseScore source page'>"
            for p in context.get("source_render_pages", [])
        )
        error_html = html.escape(
            context.get("render_error", context.get("source_window_error", ""))
        )
        if error_html:
            error_html = (
                "<p class='warning'>Source context unresolved: "
                + error_html
                + ". Crop comparison remains available; do not certify source ownership.</p>"
            )
        physical = context.get("physical_source_window")
        physical_html = (
            f" Physical source measures {physical[0] + 1}–{physical[1]}."
            if physical
            else " Source window unresolved."
        )
        system_html = (
            (
                f"<details><summary>Archived quartet system</summary><img loading='lazy' "
                f"src='{context['system_image']}'></details>"
            )
            if "system_image" in context
            else ""
        )
        cards.append(
            f"<article data-score='{context['score']}'><h2>{html.escape(context['title'])}</h2>"
            f"<p>Scan page {context['cases'][0]['page']}, detector system {context['cases'][0]['system']}. "
            f"Historical source measure labels: {measures}. {len(context['cases'])} changed crops.</p>"
            + f"<p>{physical_html}</p>"
            + "".join(comparisons)
            + system_html
            + f"<details><summary>Original scanned page</summary><img loading='lazy' src='{context['scan']}'></details>"
            + f"<details><summary>Canonical source window (MuseScore; all rendered pages)</summary>{render_html}{error_html}</details>"
            + "</article>"
        )
    options = "".join(f"<option>{score}</option>" for score in sorted({r["score"] for r in rows}))
    page = (
        """<!doctype html><meta charset='utf-8'><title>OSSQ crop differences</title>
<style>body{font:16px system-ui;max-width:1500px;margin:24px auto;background:#eef1f5;color:#18202a;padding:12px}
article{background:white;border:1px solid #ccd2d9;padding:20px;margin:24px 0;border-radius:8px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0;padding:10px;background:#f5f5f5}
figcaption{font-weight:600;margin-bottom:8px}img{max-width:100%;height:auto}section{margin:20px 0}
details{padding:12px;border:1px solid #ccd2d9;margin:12px 0}summary{cursor:pointer;font-weight:600}
details img{display:block;margin:16px auto}select{font:inherit;padding:8px}article[hidden]{display:none}
@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>OSSQ crop differences: 150 validation + 13 Borodin training crops</h1>
<p>Compare visible music: missing or clipped notes, accidentals, stems, beams and staff boundaries.
Small size/scale changes alone are not a label error. Click crops to open full resolution.
This compares two builds of the same scan, not two recognition models.</p>
<p>Source windows follow recovered historical mappings and are not yet independently certified.
MuseScore can reflow systems; all rendered PDF pages are shown. No per-case decisions are required.</p>
<label>Score: <select id='filter'><option value='all'>All scores</option>"""
        + options
        + """</select></label>
<script>document.addEventListener('DOMContentLoaded',()=>document.querySelector('#filter').addEventListener('change',e=>{
document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.score!==e.target.value)}));</script>
"""
        + "\n".join(cards)
    )
    (args.out / "index.html").write_text(page)
    diagnostics = args.recovery / "source-accidental-discrepancies.json"
    if diagnostics.exists():
        annotate_source_discrepancies(args.out, diagnostics)
    receipt = dict(
        crops=len(rows),
        systems=len(prepared),
        rendered_systems=sum(bool(c.get("source_render_pages")) for c in prepared),
        render_errors=sum("render_error" in c for c in prepared),
        unresolved_source_windows=sum("source_window_error" in c for c in prepared),
        certified_scan_ownership=False,
        systems_with_context=prepared,
        asset_sha256={
            str(p.relative_to(args.out)): sha(p) for p in assets.iterdir() if p.is_file()
        },
    )
    if diagnostics.exists():
        receipt["source_diagnostics_sha256"] = sha(diagnostics)
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in receipt.items() if k not in ("systems_with_context", "asset_sha256")}
        )
    )


if __name__ == "__main__":
    main()
