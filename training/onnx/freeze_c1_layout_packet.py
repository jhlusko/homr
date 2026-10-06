"""Freeze new calibration layouts without OCR or model prediction selection."""

import argparse
import html
import json
from pathlib import Path

from PIL import Image

from training.onnx.qualify_joint_export import sha, write


def choose(eligible, per_stratum=2):
    if per_stratum < 1:
        raise ValueError("At least one page per stratum is required")
    ordered = sorted(eligible, key=lambda r: r["page"]["pixel_sha256"])
    chosen, used = [], set()
    for stratum in ("multi-opening", "multi-continuation", "single-opening", "single-continuation"):
        count = 0
        for item in ordered:
            if item["stratum"] == stratum and item["page"]["score"] not in used:
                chosen.append(item)
                used.add(item["page"]["score"])
                count += 1
                if count == per_stratum:
                    break
    for item in ordered:
        if len(chosen) >= 4 * per_stratum:
            break
        if item["page"]["score"] not in used:
            chosen.append(item)
            used.add(item["page"]["score"])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pages", "alignment", "lyrics", "source-audit", "existing", "out", "review"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=2)
    args = parser.parse_args()
    if args.out.exists() or args.review.exists():
        raise ValueError("Outputs must be new")
    pages = [json.loads(line) for line in args.pages.read_text().splitlines()]
    alignment = json.loads(args.alignment.read_text())
    if alignment["model_predictions_used"] is not False:
        raise ValueError("Independent alignment required")
    lyrics = [json.loads(line) for line in args.lyrics.read_text().splitlines()]
    audit = json.loads(args.source_audit.read_text())
    if audit["pages_sha256"] != sha(args.pages):
        raise ValueError("Source audit partition mismatch")
    existing = {p["score"] for p in json.loads(args.existing.read_text())}
    eligible = []
    for page in pages:
        if (
            page["partition"] != "calibration"
            or page["score"] in existing
            or page["width"] * page["height"] > 8_000_000
        ):
            continue
        systems = [
            s
            for s in alignment["scores"].get(page["score"], {}).get("systems", [])
            if Path(s["page_image"]).name == Path(page["image"]).name
        ]
        systems.sort(key=lambda s: s["system_index"])
        if (
            not systems
            or any(
                s.get("status") != "aligned"
                or s.get("boundary_certified") is not True
                or s.get("count_exact") is not True
                for s in systems
            )
            or any(a["end_measure"] != z["start_measure"] for a, z in zip(systems, systems[1:]))
        ):
            continue
        start, end = systems[0]["start_measure"], systems[-1]["end_measure"]
        targets = [
            r
            for r in lyrics
            if r["score"] == page["score"] and start < r["physical_measure"] <= end
        ]
        source = audit["scores"].get(page["score"])
        if len(targets) < 10 or source is None:
            continue
        verses = sorted({r["verse_number"] for r in targets})
        eligible.append(
            {
                "page": page,
                "systems": systems,
                "start_measure": start,
                "end_measure": end,
                "native_verses": verses,
                "native_targets": len(targets),
                "source": source,
                "stratum": (
                    ("multi" if len(verses) >= 2 else "single")
                    + ("-opening" if start == 0 else "-continuation")
                ),
            }
        )
    chosen = choose(eligible, args.per_stratum)
    args.out.mkdir(parents=True)
    args.review.mkdir(parents=True)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Additional lyric layouts</title>',
        "<style>body{font:17px system-ui;max-width:1300px;margin:30px auto}",
        "img{max-width:100%}article{margin:35px 0;border-top:1px solid #aaa}</style>",
        "<h1>Frozen additional calibration layouts</h1>",
        "<p>Selection uses independent native boundaries, verse count and pixel hash. ",
        "No OCR or model predictions. Edition identity, verse order, row transcription ",
        "and note pixels still need manual certification. These are calibration pages.</p>",
    ]
    for i, item in enumerate(chosen):
        page = item["page"]
        source_path = Path(item["source"]["source"])
        if not source_path.is_absolute():
            source_path = Path(__file__).resolve().parents[3] / source_path
        if (
            sha(page["image"]) != page["image_sha256"]
            or sha(source_path) != item["source"]["source_sha256"]
        ):
            raise ValueError("Frozen source/image changed")
        item["visual_certification"] = "pending"
        with Image.open(page["image"]) as image:
            image.thumbnail((1300, 1900))
            name = f"page-{i:02}.png"
            image.save(args.review / name)
        parts.append(
            f'<article><h2>{html.escape(page["score"])} — {item["stratum"]}</h2>'
            f'<p>{html.escape(Path(page["image"]).name)}; physical measures '
            f'{item["start_measure"]+1}–{item["end_measure"]}; '
            f'{item["native_targets"]} native lyric entries; '
            f'verses {html.escape(", ".join(item["native_verses"]))}</p>'
            f'<p>Native source: {html.escape(item["source"]["source"])}</p>'
            f'<a href="{name}"><img src="{name}"></a></article>'
        )
    write(args.out / "selection.json", chosen)
    write(
        args.out / "receipt.json",
        {
            "source_sha256": sha(Path(__file__)),
            "inputs": {
                str(p): sha(p)
                for p in (args.pages, args.alignment, args.lyrics, args.source_audit, args.existing)
            },
            "selection_sha256": sha(args.out / "selection.json"),
            "eligible": len(eligible),
            "selected": len(chosen),
            "partition": "calibration",
            "selection": (
                f"{args.per_stratum} distinct scores per verse/opening stratum; hash-order fill to {4 * args.per_stratum}"
            ),
            "model_predictions_used": False,
            "OCR_used": False,
            "manual_certification": "pending; no quality score or note attachment",
        },
    )
    (args.review / "index.html").write_text("\n".join(parts))
    print(
        json.dumps(
            [
                {"score": r["page"]["score"], "stratum": r["stratum"], "verses": r["native_verses"]}
                for r in chosen
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
