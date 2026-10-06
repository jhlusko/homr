"""Build a local side-by-side OSSQ page-alignment review."""

import argparse
import html
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())
    cards = []
    for index, page in enumerate(selection):
        folder = args.selection.parent / page["score"]
        scan = args.out / f"{index:02}-{page['score']}-scan.png"
        preview = args.out / f"{index:02}-{page['score']}-source-full.png"
        shutil.copyfile(page["image"], scan)
        shutil.copyfile(folder / "source-full.png", preview)
        counts = [s["end_measure"] - s["start_measure"] for s in page["systems"]]
        systems = ", ".join(str(x) for x in counts)
        table = "".join(
            f"<tr><td>{i + 1}</td><td>{s['start_measure'] + 1}–{s['end_measure']}</td><td>{s['end_measure'] - s['start_measure']}</td></tr>"
            for i, s in enumerate(page["systems"])
        )
        cards.append(
            f"<article><h2>{html.escape(page['score'])}</h2>"
            f"<p>Proposed opening window: measures {page['proposed_start_measure'] + 1}–{page['proposed_end_measure']} "
            f"(source indices {page['proposed_start_measure']}–{page['proposed_end_measure'] - 1}). "
            f"Canonical source system measure counts: {html.escape(systems)}.</p>"
            "<table><tr><th>System</th><th>Source measures</th><th>Count</th></tr>"
            + table
            + "</table>"
            f"<div class='pair'><figure><figcaption>Scanned page</figcaption><img src='{scan.name}'></figure>"
            f"<figure><figcaption>MuseScore source preview</figcaption><img src='{preview.name}'></figure></div>"
            "<p class='check'>Review: same edition, four-part order, and system boundaries/counts.</p></article>"
        )
    page = """<!doctype html><meta charset='utf-8'><title>OSSQ crop alignment review</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;background:#f4f6f8;color:#18202a}
article{background:#fff;border:1px solid #ccd2d9;border-radius:8px;padding:20px;margin:24px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}img{max-width:100%;height:auto;border:1px solid #aaa}
figcaption{font-weight:700;margin-bottom:8px}table{border-collapse:collapse;margin:10px 0}td,th{border:1px solid #bbc;padding:6px 10px}.check{background:#fff4d6;padding:14px}
@media(max-width:900px){.pair{grid-template-columns:1fr}}</style>
<h1>OSSQ page and crop-alignment review</h1>
<p>These four pages are frozen diagnostic inputs. The source preview is generated
from the canonical MusicXML with MuseScore; the scan is the proposed opening page.
MuseScore may reflow systems for its own page width; its visual line breaks are not
asserted as scan boundaries. The table is the canonical source boundary metadata.
No model predictions are used. Review page identity, four-part order, and system
measure counts only. This is not a note-level annotation review.</p>
""" + "\n".join(cards)
    (args.out / "index.html").write_text(page)
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
