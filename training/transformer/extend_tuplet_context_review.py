# ruff: noqa: E501
"""Add recovered page context and original full-page links to the nine-case review."""

import argparse
import html
import json
import re
import shutil
from pathlib import Path

from training.transformer.audit_cached_joint_decoding import sha


def declaration_text(record):
    if record is None:
        return "None emitted before this crop on the same page"
    name = Path(record["id"]).name
    match = re.fullmatch(r"sq\d+_(\d+)_(\d+)_\d+\.txt", name)
    where = f"page {int(match[1])}, system {int(match[2])}" if match else name
    beats = record["beats"] if record["beats"] is not None else "unknown numerator"
    return f"{beats}/{record['denominator']} · {where}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((args.review / "cases.json").read_text())
    recovered = json.loads((args.context / "cases.json").read_text())
    negative = {r["id"]: r for r in recovered if r["kind"] == "false_tuplets_fully_repaired"}
    groups = {
        tuple(r["group"]): r for r in json.loads((args.context / "page-voices.json").read_text())
    }
    document = (args.review / "index.html").read_text()
    if 'data-recovered-context="1"' in document:
        raise ValueError("Review already extended")
    if len(cases) != 9 or len(negative) != 9:
        raise ValueError("Missing nine-case context coverage")
    for case in cases:
        row = negative[case["id"]]
        page = groups[tuple(row["group"])]["original_page"]
        source = Path(page["path"])
        if sha(source) != page["sha256"]:
            raise ValueError("Original page changed")
        target = args.review / "assets" / source.name
        shutil.copyfile(source, target)
        count = len(row["same_page_systems"])
        old = row["note_rest_metrics"]["standard"]["symbol"]
        guard = row["note_rest_metrics"]["guard"]["symbol"]
        prior = declaration_text(row["latest_prior_page_meter"])
        same = declaration_text(row["latest_same_page_meter"])
        body = (
            '<aside class="notice" data-recovered-context="1"><b>Recovered page/voice context</b>'
            f'<p>{count} neighboring systems for part {row["group"][2]}, page {row["group"][1]}. '
            f'<a href="assets/{html.escape(target.name)}" target="_blank">Open the original full scan page</a>.</p>'
            f"<p>Latest same-page declaration: <b>{html.escape(same)}</b>.<br>"
            f"Latest earlier-page declaration: <b>{html.escape(prior)}</b> "
            "(diagnostic evidence; cross-page meter is not currently carried by HOMR).</p>"
            f"<p>After joining the cached systems, current cleanup has <b>{old}</b> note/rest errors "
            f"in this crop; the guard has <b>{guard}</b>. These differ from the isolated-crop panels below.</p>"
            '<p class="small">Published-crop reconstruction, not a new production segmentation run. '
            "Note/rest scores omit repeated key-signature headers; source XML voice ownership and "
            "OSSQ native measure boundaries are not independently certified.</p></aside>"
        )
        marker = f'<article id="{case["key"]}">'
        begin = document.index(marker)
        insertion = document.index("</header>", begin) + len("</header>")
        document = document[:insertion] + body + document[insertion:]
        case["recovered_context"] = row
    (args.review / "index.html").write_text(document)
    (args.review / "cases-with-context.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.review / "context-provenance.json").write_text(
        json.dumps(
            {
                "tool_sha256": sha(__file__),
                "inputs": {
                    str(p): sha(p)
                    for p in [args.context / "cases.json", args.context / "page-voices.json"]
                },
                "extended_review_sha256": sha(args.review / "index.html"),
                "cases": 9,
            },
            indent=2,
        )
        + "\n"
    )
    print("Extended nine review cases with context")


if __name__ == "__main__":
    main()
