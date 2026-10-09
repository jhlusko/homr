"""Render native-system windows beside scans; no prediction-derived reference links."""

import argparse
import html
import json
import os
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from PIL import Image, ImageChops

from training.onnx.audit_full_page_references import read_mxl
from training.onnx.prepare_page_certification_extension import preview_window


def sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--musescore", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("Render concurrency is bounded at two")
    selected_path = args.references / "candidate-pages.json"
    pages = json.loads(selected_path.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    runtime = args.out / "qt-runtime"
    runtime.mkdir(mode=0o700)
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="offscreen", XDG_RUNTIME_DIR=str(runtime))
    fallback = args.musescore.parent.parent / "fallback"
    libraries = sorted({str(p.parent) for p in fallback.rglob("*.so*") if p.is_file()})
    env["LD_LIBRARY_PATH"] = ":".join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    version = subprocess.run(
        [str(args.musescore), "--version"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    prepared, hashes = [], {str(selected_path): sha(selected_path)}
    jobs = []
    for page in pages:
        source, scan = Path(page["native_source"]), Path(page["image"])
        if sha(source) != page["native_source_sha256"] or sha(scan) != page["image_sha256"]:
            raise ValueError("Frozen source or scan differs")
        folder = args.out / page["score"]
        folder.mkdir()
        shutil.copyfile(scan, folder / "scan-full.png")
        root = read_mxl(source)
        metadata = args.workspace / f"homr-artifacts/lieder-v8/_build/systems/{page['score']}.yaml"
        hashes[str(metadata)] = sha(metadata)
        document = yaml.safe_load(metadata.read_text())
        candidates = [p for p in document["pages"].values() if Path(p["image"]).name == scan.name]
        if len(candidates) != 1:
            raise ValueError("Scan geometry identity ambiguous")
        geometry = candidates[0]
        with Image.open(scan) as pixels:
            crop_geometry_verified = list(pixels.size) == [geometry["width"], geometry["height"]]
            if len(geometry["systems"]) != len(page["systems"]):
                raise ValueError("Scan system coverage differs")
            systems = []
            for index, (bounds, detected) in enumerate(zip(page["systems"], geometry["systems"])):
                name = f"system-{index+1:02}"
                box = detected["boundingBox"]
                crop_box = [
                    max(0, box["left"] - 25),
                    max(0, box["top"] - 40),
                    min(pixels.width, box["left"] + box["width"] + 25),
                    min(pixels.height, box["top"] + box["height"] + 40),
                ]
                if crop_geometry_verified:
                    pixels.crop(crop_box).save(folder / (name + "-scan.png"))
                preview = preview_window(root, bounds["start_measure"], bounds["end_measure"])
                # Carry attributes for display; raw native source stays authoritative.
                xml = folder / (name + "-source.musicxml")
                ET.ElementTree(preview).write(xml, encoding="utf-8", xml_declaration=True)
                system = {
                    "index": index,
                    "start_measure": bounds["start_measure"],
                    "end_measure": bounds["end_measure"],
                    "cropBox": crop_box,
                    "scan": str(
                        (
                            folder
                            / (name + "-scan.png" if crop_geometry_verified else "scan-full.png")
                        ).relative_to(args.out)
                    ),
                    "cropGeometryVerified": crop_geometry_verified,
                    "cropWarning": (
                        ""
                        if crop_geometry_verified
                        else "Full page shown: cached system coordinates use different dimensions."
                    ),
                    "xml": str(xml.relative_to(args.out)),
                    "sourcePages": [],
                    "status": "pending",
                }
                systems.append(system)
                jobs.append((folder, xml, system))
        prepared.append(
            dict(
                page,
                systems=systems,
                fullScan=str((folder / "scan-full.png").relative_to(args.out)),
            )
        )
    lock = threading.Lock()
    done = 0
    started = time.monotonic()

    def render(job):
        nonlocal done
        folder, xml, system = job
        pdf = xml.with_suffix(".pdf")
        try:
            with xml.with_suffix(".log").open("w") as log:
                subprocess.run(
                    [str(args.musescore), "-o", str(pdf), str(xml)],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=True,
                )
                subprocess.run(
                    ["pdftoppm", "-scale-to", "1800", "-png", str(pdf), str(xml.with_suffix(""))],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=120,
                    check=True,
                )
            for image in sorted(folder.glob(xml.stem + "-*.png")):
                with Image.open(image) as pixels:
                    rgb = pixels.convert("RGB")
                    ink = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
                    if ink is None:
                        raise ValueError("Empty source render")
                    crop = [
                        max(0, ink[0] - 20),
                        max(0, ink[1] - 20),
                        min(rgb.width, ink[2] + 20),
                        min(rgb.height, ink[3] + 20),
                    ]
                    rgb.crop(crop).save(image)
                system["sourcePages"].append(str(image.relative_to(args.out)))
            if not system["sourcePages"]:
                raise ValueError("No source preview pages")
            system["status"] = "rendered"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            system.update(status="failed", error=str(error))
        with lock:
            done += 1
            progress = {
                "completed": done,
                "total": len(jobs),
                "seconds": time.monotonic() - started,
            }
            (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(progress), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(render, jobs))
    cards = []
    for page in prepared:
        cards.append(
            f'<article><h2>{html.escape(page["score"])}</h2>'
            f'<p>Native physical measures {page["start_measure"]+1}–{page["end_measure"]}; '
            f'{page["page_native_lyric_targets"]} lyric targets; '
            f'{len(page["source_lyric_parts"])} lyric-bearing parts. '
            f'<a href="{page["fullScan"]}">Full original scan</a></p>'
        )
        for system in page["systems"]:
            source_images = "".join(
                f'<a href="{image}"><img src="{image}" alt="Native source fragment"></a>'
                for image in system["sourcePages"]
            )
            error = html.escape(system.get("error", ""))
            warning = html.escape(system["cropWarning"])
            cards.append(
                f'<h3>Scan system {system["index"]+1}: proposed native measures '
                f'{system["start_measure"]+1}–{system["end_measure"]}</h3><p>{warning}</p>'
                f'<div class="pair"><figure><figcaption>Original scan context</figcaption>'
                f'<a href="{system["scan"]}"><img src="{system["scan"]}" '
                'alt="Original scan system"></a>'
                "</figure><figure><figcaption>Native MusicXML fragment — "
                "regenerated layout</figcaption>"
                f"{source_images}<p>{error}</p></figure></div>"
            )
        cards.append("</article>")
    markup = (
        '<!doctype html><meta charset="utf-8"><title>Native lyric linkage</title>'
        "<style>body{font:17px system-ui;margin:2rem;background:#f5f6f7}"
        "article{background:white;padding:1rem;margin:2rem 0}.pair{display:grid;"
        "grid-template-columns:1fr 1fr;gap:1rem}img{max-width:100%}"
        "figure{margin:0}figcaption{font-weight:bold}@media(max-width:900px)"
        "{.pair{grid-template-columns:1fr}}</style><h1>Native lyric page/system linkage</h1>"
        "<p>Ten fixed new review cases, not untouched model tests. No recognizer output "
        "is used. Compare passage content, lyrics and first/last measure anchors. "
        "Proposed measure ranges come from count alignment, not certified note pixels. "
        "MuseScore regenerates layout; its line breaks do not prove scan boundaries. "
        "Source fragment attributes are inherited for display. Scan context can include "
        "neighbouring material; full scans are available. All rendered pages are shown. "
        "Raw native MusicXML remains authoritative.</p>"
    )
    (args.out / "index.html").write_text(markup + "\n".join(cards))
    report = {
        "pages": prepared,
        "inputSha256": hashes,
        "sourceSha256": sha(Path(__file__)),
        "musescoreSha256": sha(args.musescore),
        "musescoreVersion": version.stdout.strip(),
        "renders": len(jobs),
        "renderFailures": sum(s["status"] == "failed" for _, _, s in jobs),
        "model_predictions_used": False,
        "pixelLinksCertified": False,
        "scope": (
            "Visual passage/edition linkage proposals only; "
            "no automatic certification or accuracy claim."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({"renders": report["renders"], "failures": report["renderFailures"]}), flush=True
    )


if __name__ == "__main__":
    main()
