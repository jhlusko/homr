"""Original-scan and CTC-position side-by-side calibration review."""

import argparse
import html
import json
from fractions import Fraction
from pathlib import Path

from PIL import Image

SELECTION = {
    "IMSLP154060": [16, 18, 20, 21, 22, 30, 31, 37],
    "IMSLP89794": [32, 36, 43, 58, 59, 89, 90, 94],
}


def location(row):
    return f'measure {row["physical_measure"]}, beat {Fraction(row["onset_quarters"])+1}'


def build(source, out):
    out.mkdir(parents=True, exist_ok=False)
    scores = json.loads((source / "grouped-score.json").read_text())
    cases = {r["box_index"]: r for r in scores["cases"]}
    cards = []
    for score, indices in SELECTION.items():
        data = json.loads((source / (score + ".json")).read_text())
        image = Image.open(data["page"]["image"]).convert("RGB")
        for index in indices:
            row = data["rows"][index]
            x = [p[0] for p in row["box"]]
            y = [p[1] for p in row["box"]]
            bounds = [
                max(0, int(min(x)) - 30),
                max(0, int(min(y)) - 160),
                min(image.width, int(max(x)) + 30),
                min(image.height, int(max(y)) + 45),
            ]
            name = f"{score}-{index}.png"
            image.crop(bounds).save(out / name)
            width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
            marks, labels = [], []
            for syllable in row["syllables"]:
                left, top, right, bottom = syllable["bounds"]
                marks.append(
                    f'<rect x="{left-bounds[0]}" y="{top-bounds[1]}" width="{right-left}" height="{bottom-top}" fill="#16a34a" fill-opacity=".08" stroke="#16a34a" stroke-width="1.5"><title>{html.escape(syllable["text"])}</title></rect>'
                )
                cx, cy = syllable["center"]
                marks.append(
                    f'<circle cx="{cx-bounds[0]}" cy="{cy-bounds[1]}" r="3" fill="#d97706"/>'
                )
            case = cases.get(index) if score == "IMSLP154060" else None
            if case:
                for ref, attention, detected in zip(
                    case["reference"],
                    case["predictions"]["attention"],
                    case["predictions"]["detected_noteheads"],
                    strict=True,
                ):
                    wrong = location(attention) != location(ref)
                    labels.append(
                        f'<li><strong>{html.escape(detected["text"])}</strong>: native “{html.escape(ref["text"])}”, {location(ref)}; <span class="{"wrong" if wrong else ""}">attention {location(attention)}</span>; notehead {location(detected)}.</li>'
                    )
                scope = "Conditional comparison: system and verse fixed from scan review. Source ownership is a calibration linkage proposal."
            else:
                scope = "Geometry review only; this box has no scored native attachment reference."
            caption = " | ".join(s["text"] for s in row["syllables"])
            cards.append(
                f'<section><h2>{html.escape(score)} — “{html.escape(row["text"])}”</h2><p>{scope}</p><div class="panels"><figure><figcaption>Original scan; cached full-line text</figcaption><img src="{name}" alt="Original lyric and note context"></figure><figure><figcaption>CTC-positioned syllable candidates</figcaption><svg viewBox="0 0 {width} {height}"><image href="{name}" width="{width}" height="{height}"/>{"".join(marks)}</svg></figure></div><p>{html.escape(caption)}</p><ul>{"".join(labels)}</ul></section>'
            )
    (out / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Grouped lyric positions</title><style>body{font:16px system-ui;max-width:1450px;margin:25px auto;padding:0 16px;background:#f3f4f6;color:#111827}section{background:white;padding:18px;margin:22px 0}.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img,svg{width:100%}figcaption{margin-bottom:8px}.wrong{color:#b91c1c;font-weight:600}@media(max-width:850px){.panels{grid-template-columns:1fr}}</style>
<h1>Grouped lyric positions</h1><p>Same full-line OCR text; syllables split at recognized spaces/hyphens using approximate CTC character positions. Green: syllable bounds; orange: recognized-character mean position. No source lyric prompt or equal-width split.</p>
<p>All149 cached text regions and parent boxes reproduced exactly. Six selected boxes:17 native syllables. Attention anchors attach14/17 correctly; detected-notehead anchors17/17. Text is correct15/17 under literal native matching. This uses fixed system/verse routing and count-exact vocal source linkage; it does not qualify automatic lyrics fusion. Additional examples are unscored geometry controls.</p>"""
        + "".join(cards)
    )
    return len(cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.out), "review cases")
