"""Show diagnostic head/attention differences without assigning correctness labels."""

import argparse
import html
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    keys = {
        k["id"]: k
        for k in json.loads(
            (
                args.base
                / "guarded-tuplet-20261004/review/lyric-anchor-judgement-2-20261008-key.json"
            ).read_text()
        )
    }
    pages = dict(
        line.split(" ", 1)
        for line in (
            (args.base / "lyric-fresh2-upload-audit-20261008/fresh2-pages.txt")
            .read_text()
            .splitlines()
        )
    )
    targets = [dict(t, score=r["score"]) for r in report["records"] for t in r["targets"]]
    matched = sorted(
        (t for t in targets if t["status"] == "forced_sequence_match"),
        key=lambda t: -t["displacementPixels"],
    )
    selected = matched[:20]
    args.out.mkdir(parents=True, exist_ok=False)
    cards = []
    for target in selected:
        key = keys[target["id"]]
        image = cv2.imread(pages[key["score"]])
        scale = (
            image.shape[1]
            / json.loads(
                (
                    args.base
                    / "lyric-fresh2-upload-audit-20261008/saved-fresh2-inputs"
                    / f"review-{key['score']}.json"
                ).read_text()
            )["raster"]["w"]
        )
        x, y = target["detectedX"] * scale, target["detectedY"] * scale
        attention = target["attentionX"] * scale
        lyric = target["lyricX"] * scale
        box = key["box"]
        left = max(0, int(min(x, attention, lyric) - 180))
        right = min(image.shape[1], int(max(x, attention, lyric) + 180))
        top = max(0, int(min(y - 70, box[1] - 20)))
        bottom = min(image.shape[0], int(max(y + 50, box[3] + 20)))
        crop = image[top:bottom, left:right].copy()
        cv2.circle(crop, (round(x - left), round(y - top)), 12, (0, 140, 0), 3)
        cv2.line(
            crop,
            (round(attention - left), 0),
            (round(attention - left), crop.shape[0] - 1),
            (255, 100, 0),
            2,
        )
        mid = (box[1] + box[3]) / 2 - top
        cv2.line(
            crop,
            (round(lyric - left), round(mid) - 10),
            (round(lyric - left), round(mid) + 10),
            (0, 128, 255),
            3,
        )
        name = target["id"] + ".png"
        cv2.imwrite(str(args.out / name), crop)
        cards.append(
            f'<article><h2>{html.escape(key["score"])} — '
            f'{html.escape(target["text"])}</h2>'
            f'<p>Detected head vs old attention: {target["displacementPixels"]:.1f} '
            f'raster pixels. Printed line {key["printedLine"]}. '
            "Development sequence match; correctness unconfirmed.</p>"
            f'<img src="{name}" alt="Green detected head, blue old attention '
            'and orange old text estimate"></article>'
        )
    markup = (
        '<!doctype html><meta charset="utf-8"><title>Lyric geometry audit</title>'
        "<style>body{font:17px system-ui;max-width:1200px;margin:2rem}"
        "article{border-top:1px solid #aaa;padding:1rem 0}img{max-width:100%}</style>"
        "<h1>Lyric geometry audit — development only</h1>"
    )
    markup += (
        f"<p>{len(targets)} sampled placements; {len(matched)} have a forced sequence match. "
        f"Showing the {len(selected)} largest head/attention differences. "
        "This is association coverage, not accuracy or recall. "
        "Earlier accepted judgements are provisional.</p>"
    )
    markup += (
        "<p>Green circle: associated detected notehead. Blue line: old decoder attention. "
        "Orange tick: old estimated lyric center, not a word boundary. "
        "Text shown is original OCR, including errors; "
        "these positions have not been human-confirmed. No runtime or score changes.</p>"
    )
    (args.out / "index.html").write_text(markup + "\n".join(cards))
    (args.out / "items.json").write_text(json.dumps(selected, indent=2) + "\n")
    (args.out / "audit-report.json").write_bytes(args.report.read_bytes())
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps({"targets": len(targets), "forcedMatches": len(matched), "shown": len(selected)})
    )


if __name__ == "__main__":
    main()
