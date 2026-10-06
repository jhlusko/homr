"""Build a static review page for lyric/direction role controls."""

import html
import json
from pathlib import Path


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.inventory.read_text())
    cards = []
    for case in cases:
        cards.append(
            "<article><h2>"
            + html.escape(case["score"] + " — " + case["text"])
            + "</h2><p>"
            + html.escape(case["scope"])
            + "</p><div class='pair'><figure><figcaption>Original scan crop</figcaption>"
            + f"<img src='../c1-semantic-controls-20261005/{html.escape(case['crop'])}'></figure>"
            + "<figure><figcaption>Role interpretation</figcaption>"
            + "<p class='decision'>Review visually: lyric, direction, or ambiguous."
            + " The current geometry-only route does not decide this.</p></figure></div></article>"
        )
    page = """<!doctype html><meta charset='utf-8'><title>Lyric and direction role controls</title>
<style>body{font:17px system-ui;max-width:1100px;margin:28px auto;background:#f5f6f8;color:#18202a}
article{background:white;padding:20px;margin:24px 0;border:1px solid #ccd2d9;border-radius:8px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%;height:auto}
figcaption{font-weight:700;margin-bottom:8px}.decision{padding:20px;background:#fff4d6}
@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>Lyric versus direction role controls</h1>
<p>These are real scan crops where routing alone is insufficient. Existing OCR text and
staff geometry are shown for review; no native labels or word whitelist were used in
inference. The safe production behavior is to leave ambiguous rows unresolved.</p>
""" + "\n".join(cards)
    (args.out / "index.html").write_text(page)
    (args.out / "inventory.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
