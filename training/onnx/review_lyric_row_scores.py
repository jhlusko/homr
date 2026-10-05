"""Show verified original lyric rows beside separately scored OCR predictions."""

import argparse
import html
import json
import shutil
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Review output must be new")
    references = json.loads(args.references.read_text())
    reports = {p.stem: json.loads(p.read_text()) for p in args.scores}
    if any(r["reference_sha256"] != sha(args.references) for r in reports.values()):
        raise ValueError("Scores refer to different row references")
    args.out.mkdir(parents=True)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Lyric row scoring</title>',
        "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:15px}",
        "article{border:1px solid #ccc;border-radius:8px;margin:24px 0;padding:18px}",
        "img{max-width:100%;width:auto}pre{white-space:pre-wrap;font:16px system-ui}",
        ".predictions{display:flex;gap:20px;flex-wrap:wrap}"
        ".predictions section{flex:1;min-width:250px}</style>",
        "<h1>Calibration lyric rows: scan versus OCR</h1>",
        "<p>Rows were manually transcribed from the original scans. "
        "Review was not blinded to prior OCR.",
        "Character scoring removes layout connectors and whitespace; "
        "preserves accents, apostrophes and punctuation.",
        "These are conditional row results, not note attachment or held-out release scores.</p>",
    ]
    for name, report in reports.items():
        parts.append(
            f"<p><b>{html.escape(name)}</b>: "
            f'{report["errors"]}/{report["support"]} character edits '
            f'({report["character_error_percent"]:.2f}%).</p>'
        )
    for row in references:
        name = row["id"] + ".png"
        source = args.references.parent / name
        if sha(source) != row["crop_sha256"]:
            raise ValueError("Original row crop changed")
        shutil.copyfile(source, args.out / name)
        parts.append(
            f'<article><h2>{html.escape(row["score"])} — system {row["system_index"]+1}, '
            f'native verse {html.escape(row["source_verse"])}</h2><img src="{html.escape(name)}">'
            f'<p>Scan transcription:</p><pre>{html.escape(row["visible_text"])}</pre>'
            '<div class="predictions">'
        )
        for label, report in reports.items():
            result = next(r for r in report["rows"] if r["id"] == row["id"])
            parts.append(
                f"<section><h3>{html.escape(label)}</h3>"
                f'<p>{result["errors"]}/{result["support"]} edits</p>'
                f'<pre>{html.escape(result["predicted_text"])}</pre></section>'
            )
        parts.append("</div></article>")
    (args.out / "index.html").write_text("\n".join(parts))
    write(
        args.out / "receipt.json",
        {
            "references_sha256": sha(args.references),
            "reports_sha256": {str(p): sha(p) for p in args.scores},
            "source_sha256": sha(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
