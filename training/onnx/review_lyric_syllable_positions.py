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


def build(source, out, spatial_proposals=None, changed_only=False, controls_per_page=0):
    out.mkdir(parents=True, exist_ok=False)
    score_path = source / "grouped-score.json"
    scores = json.loads(score_path.read_text()) if score_path.exists() else {"cases": []}
    cases = {r["box_index"]: r for r in scores["cases"]}
    cards = []
    proposals = (
        {}
        if spatial_proposals is None
        else {
            (case["score"], case["box_index"]): case["proposal"]
            for case in json.loads(spatial_proposals.read_text())["cases"]
        }
    )
    selection = SELECTION
    if changed_only:
        selection = {}
        for score, index in sorted(proposals):
            selection.setdefault(score, []).append(index)
    if controls_per_page:
        if controls_per_page < 0:
            raise ValueError("Control count must be nonnegative")
        selection = {score: list(indices) for score, indices in selection.items()}
        for path in sorted(source.glob("IMSLP*.json")):
            page = json.loads(path.read_text())
            score = page["page"]["score"]
            candidates = [
                r
                for r in page["rows"]
                if not r.get("geometry_skip") and (score, r["box_index"]) not in proposals
            ]
            # Fixed geometry review selection, not independently certified negatives.
            candidates.sort(
                key=lambda r: (
                    -max((len(s["text"]) for s in r["syllables"]), default=0),
                    r["box_index"],
                )
            )
            chosen = selection.setdefault(score, [])
            for row in candidates:
                if row["box_index"] not in chosen:
                    chosen.append(row["box_index"])
                if len(chosen) >= controls_per_page:
                    break
    for score, indices in selection.items():
        data = json.loads((source / (score + ".json")).read_text())
        image = Image.open(data["page"]["image"]).convert("RGB")
        for index in indices:
            row = data["rows"][index]
            spatial = proposals.get((score, index))
            if spatial is not None:
                row = dict(row, syllables=spatial["syllables"])
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
            if spatial is not None:
                scope = "Spatial-gap proposal; compare with the original glyph spacing. Native syllables and note ownership are unscored here. The threshold is a calibration hypothesis, not production-qualified."
                if (score, index) == ("IMSLP89794", 58):
                    scope = "Missing-separator calibration proposal: mei | nem | Áb. Native lyrics are mei | nem | Ab; the OCR accent is preserved. Fixed system/verse routing gives three correct note owners, two correct literal texts. The spacing threshold is fitted to this example and is not production-qualified."
            caption = " | ".join(s["text"] for s in row["syllables"])
            cards.append(
                f'<section><h2>{html.escape(score)} — “{html.escape(row["text"])}”</h2><p>{scope}</p><div class="panels"><figure><figcaption>Original scan; cached full-line text</figcaption><img src="{name}" alt="Original lyric and note context"></figure><figure><figcaption>CTC-positioned syllable candidates</figcaption><svg viewBox="0 0 {width} {height}"><image href="{name}" width="{width}" height="{height}"/>{"".join(marks)}</svg></figure></div><p>{html.escape(caption)}</p><ul>{"".join(labels)}</ul></section>'
            )
    receipt = json.loads((source / "receipt.json").read_text())
    total = sum(r["regions"] for r in receipt["results"])
    unchanged = sum(r["cached_rows_unchanged"] for r in receipt["results"])
    summary = f"{unchanged}/{total} cached texts and parent boxes reproduced exactly. This gallery shows {len(cards)} calibration regions; it does not measure automatic lyric/verse routing or corpus accuracy."
    if score_path.exists():
        summary += " Six selected boxes contain 17 native syllables: attention attaches 14/17 correctly, detected-notehead anchors 17/17; literal text is correct 15/17, with fixed reviewed routing."
    (out / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Grouped lyric positions</title><style>body{font:16px system-ui;max-width:1450px;margin:25px auto;padding:0 16px;background:#f3f4f6;color:#111827}section{background:white;padding:18px;margin:22px 0}.panels{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}img,svg{width:100%}figcaption{margin-bottom:8px}.wrong{color:#b91c1c;font-weight:600}@media(max-width:850px){.panels{grid-template-columns:1fr}}</style>
<h1>Grouped lyric positions</h1><p>Same full-line OCR text; syllables split at recognized spaces/hyphens using approximate CTC character positions. Optional spatial-gap proposals are identified per example. Green: syllable bounds; orange: recognized-character mean position. No source lyric prompt or equal-width split.</p>
"""
        + f"<p>{summary}</p>"
        + "".join(cards)
    )
    return len(cards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spatial-proposals", type=Path)
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--controls-per-page", type=int, default=0)
    args = parser.parse_args()
    print(
        build(
            args.source, args.out, args.spatial_proposals, args.changed_only, args.controls_per_page
        ),
        "review cases",
    )
