"""A local review tool for `ocr_first_text_ground_truth.py`'s output.

Serves one page per score: every confirmed lyric/dynamic match, a crop of the
scan around its own detected box (cropped on the fly - there's no pre-rendering
step here, unlike `stage2_pair_review_server.py`'s notation renders, since a plain
image crop is cheap), its OCR'd text, kind, and match confidence, each with
Good/Bad/Unclear buttons - the same systematic "validate before trusting at
scale" pass this project's other review tools exist for.

Stdlib only, matching `review_server.py`/`stage2_pair_review_server.py`'s own
house style for this project's short-lived personal review tools.
"""

# flake8: noqa: T201

import argparse
import io
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

_JUDGMENTS = ("good", "bad", "unclear")

#: Extra pixels around a match's own detected box, so the crop shows a little
#: context (the staff above a lyric line, neighboring words) rather than an
#: exact, context-free bounding box.
CROP_PADDING = 40


def load_matches(matches_dir: Path) -> list[dict]:
    """One entry per confirmed match across every `{score_id}.json` file in
    `matches_dir` - `ocr_first_text_ground_truth.py`'s own per-score output -
    annotated with a `key` unique within its own score (`"{page_index}-{index}"`,
    since matches carry no id of their own) for judgment persistence and lookup.
    """
    entries = []
    for path in sorted(matches_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        score_id = doc["score_id"]
        per_page_counter: dict[int, int] = {}
        for match in doc["matches"]:
            page_index = match["page_index"]
            index_on_page = per_page_counter.get(page_index, 0)
            per_page_counter[page_index] = index_on_page + 1
            entries.append(
                {
                    "score_id": score_id,
                    "key": f"{page_index}-{index_on_page}",
                    **match,
                }
            )
    return entries


class ReviewState:
    def __init__(self, matches_dir: Path, judgments_path: Path, pngs_dirs: list[Path]) -> None:
        self.matches_dir = matches_dir
        self.judgments_path = judgments_path
        self.pngs_dirs = pngs_dirs
        self.entries = load_matches(matches_dir)
        self.by_score: dict[str, list[dict]] = {}
        for entry in self.entries:
            self.by_score.setdefault(entry["score_id"], []).append(entry)

    def score_ids(self) -> list[str]:
        return sorted(self.by_score)

    def judgments(self) -> dict[str, dict]:
        if not self.judgments_path.exists():
            return {}
        return json.loads(self.judgments_path.read_text(encoding="utf-8"))

    def progress(self, score_id: str) -> tuple[int, int]:
        judged = self.judgments()
        entries = self.by_score.get(score_id, [])
        total = len(entries)
        done = sum(1 for e in entries if f"{score_id}/{e['key']}" in judged)
        return done, total

    def save_judgment(self, judgment_id: str, judgment: str, note: str) -> None:
        if judgment not in _JUDGMENTS:
            raise ValueError(f"unknown judgment {judgment!r}")
        current = self.judgments()
        current[judgment_id] = {"judgment": judgment, "note": note}
        self.judgments_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    def page_path(self, score_id: str, page_image: str) -> Path | None:
        for pngs_dir in self.pngs_dirs:
            candidate = pngs_dir / score_id / page_image
            if candidate.is_file():
                return candidate
        return None

    def crop(self, score_id: str, entry: dict) -> bytes | None:
        page_path = self.page_path(score_id, entry["page_image"])
        if page_path is None:
            return None
        box = entry["box"]
        with Image.open(page_path) as image:
            left = max(0, box["left"] - CROP_PADDING)
            top = max(0, box["top"] - CROP_PADDING)
            right = min(image.width, box["left"] + box["width"] + CROP_PADDING)
            bottom = min(image.height, box["top"] + box["height"] + CROP_PADDING)
            cropped = image.convert("L").crop((left, top, right, bottom))
            buffer = io.BytesIO()
            cropped.save(buffer, format="PNG")
            return buffer.getvalue()


def _safe_score_id(raw: str) -> str | None:
    score_id = unquote(raw)
    return score_id if re.match(r"^[A-Za-z0-9_-]+$", score_id) else None


INDEX_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage 3 text review</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 4px 12px; border-bottom: 1px solid #ddd; text-align: left; }}
.done {{ color: #2a7a2a; }}
.partial {{ color: #a06a00; }}
.none {{ color: #888; }}
</style></head><body>
<h1>Stage 3 text review</h1>
<p>{total_matches} matches across {total_scores} scores.</p>
<table><tr><th>Score</th><th>Reviewed</th></tr>
{rows}
</table>
</body></html>"""

SCORE_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{score_id} - Stage 3 text review</title>
<style>
body {{ font-family: sans-serif; margin: 1.5em; }}
.match {{ border: 1px solid #ccc; border-radius: 6px; padding: 12px; margin-bottom: 12px; }}
.match.good {{ border-color: #2a7a2a; background: #f3fbf3; }}
.match.bad {{ border-color: #b02a2a; background: #fdf3f3; }}
.match.unclear {{ border-color: #a06a00; background: #fdf9ee; }}
.match img {{ max-width: 100%; display: block; margin-bottom: 6px; }}
.meta {{ font-family: monospace; margin: 4px 0; }}
.kind-lyric {{ color: #2a4a9a; }}
.kind-dynamic {{ color: #9a5a2a; }}
button {{ margin-right: 6px; padding: 4px 10px; }}
nav a {{ margin-right: 16px; }}
</style></head><body>
<nav><a href="{base_path}/">&larr; all scores</a>{next_link}</nav>
<h1>{score_id}</h1>
<p>{count} matches.</p>
{matches}
<script>
function judge(id, judgment) {{
  fetch('{base_path}/api/judge', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{id: id, judgment: judgment}}),
  }}).then(() => {{
    const el = document.getElementById('match-' + id.replace('/', '_'));
    el.className = 'match ' + judgment;
  }});
}}
</script>
</body></html>"""

MATCH_TEMPLATE = """<div class="match {css_class}" id="match-{key_id}">
<img src="{base_path}/crop/{score_id}/{key}" alt="{key}">
<div class="meta"><span class="kind-{kind}">{kind}</span> "{text}" (confidence {confidence:.2f}, page {page_index})</div>
<button onclick="judge('{judgment_id}', 'good')">Good</button>
<button onclick="judge('{judgment_id}', 'bad')">Bad</button>
<button onclick="judge('{judgment_id}', 'unclear')">Unclear</button>
<span>{current_judgment}</span>
</div>"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_index(state: ReviewState, base_path: str = "") -> str:
    """The score-list landing page. `base_path` is `""` when this module runs
    standalone (its own routes are already rooted at `/`) or a mount prefix like
    `"/text"` when embedded in another review server's process - every internal
    link is built from it so the page works unmodified either way."""
    rows = []
    for score_id in state.score_ids():
        done, total = state.progress(score_id)
        css = "done" if total and done == total else ("partial" if done else "none")
        rows.append(
            f'<tr><td><a href="{base_path}/score/{score_id}">{score_id}</a></td>'
            f'<td class="{css}">{done}/{total}</td></tr>'
        )
    return INDEX_TEMPLATE.format(
        rows="\n".join(rows) or "<tr><td>none found</td></tr>",
        total_matches=len(state.entries),
        total_scores=len(state.by_score),
    )


def render_score_page(state: ReviewState, score_id: str, base_path: str = "") -> str | None:
    """One score's full match list, or `None` if `score_id` isn't known - the
    caller decides how to turn that into a 404."""
    if score_id not in state.by_score:
        return None
    judged = state.judgments()
    matches_html = []
    for entry in state.by_score[score_id]:
        judgment_id = f"{score_id}/{entry['key']}"
        record = judged.get(judgment_id)
        css_class = record["judgment"] if record else ""
        current_judgment = f"(marked {record['judgment']})" if record else ""
        confidence = entry.get("matched_fraction", entry.get("match_ratio", 0.0))
        matches_html.append(
            MATCH_TEMPLATE.format(
                score_id=score_id, key=entry["key"], key_id=entry["key"].replace("/", "_"),
                kind=entry["kind"], text=_escape(entry["text"]), confidence=confidence,
                page_index=entry["page_index"], judgment_id=judgment_id,
                css_class=css_class, current_judgment=current_judgment, base_path=base_path,
            )
        )
    all_ids = state.score_ids()
    next_link = ""
    if score_id in all_ids:
        position = all_ids.index(score_id)
        if position + 1 < len(all_ids):
            next_link = (
                f' <a href="{base_path}/score/{all_ids[position + 1]}">next score &rarr;</a>'
            )
    return SCORE_TEMPLATE.format(
        score_id=score_id,
        count=len(state.by_score[score_id]),
        matches="\n".join(matches_html),
        next_link=next_link,
        base_path=base_path,
    )


def crop_bytes(state: ReviewState, score_id: str, key: str) -> bytes | None:
    """The on-the-fly crop for one match, or `None` if the match or its scan page
    can't be found."""
    entry = next((e for e in state.by_score.get(score_id, []) if e["key"] == key), None)
    if entry is None:
        return None
    return state.crop(score_id, entry)


class Handler(BaseHTTPRequestHandler):
    state: ReviewState  # set on the class before serving

    def _send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # a personal local tool - the default per-request stderr line is just noise

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._index()
        elif path.startswith("/score/"):
            self._score_page(path.removeprefix("/score/"))
        elif path.startswith("/crop/"):
            self._crop(path.removeprefix("/crop/"))
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_html("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/judge":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        judgment_id = body.get("id", "")
        judgment = body.get("judgment", "")
        note = body.get("note", "")
        score_id, _, key = judgment_id.partition("/")
        if not any(e["score_id"] == score_id and e["key"] == key for e in self.state.entries):
            self._send_json({"error": "unknown match"}, status=404)
            return
        try:
            self.state.save_judgment(judgment_id, judgment, note)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        self._send_json({"ok": True})

    def _index(self) -> None:
        self._send_html(render_index(self.state))

    def _score_page(self, raw_id: str) -> None:
        score_id = _safe_score_id(raw_id)
        page = render_score_page(self.state, score_id) if score_id else None
        if page is None:
            self._send_html("unknown score id", status=404)
            return
        self._send_html(page)

    def _crop(self, raw_path: str) -> None:
        parts = raw_path.split("/", 1)
        if len(parts) != 2:
            self._send_html("bad path", status=400)
            return
        score_id = _safe_score_id(parts[0])
        key = unquote(parts[1])
        if score_id is None:
            self._send_html("bad score id", status=400)
            return
        entry = next(
            (e for e in self.state.by_score.get(score_id, []) if e["key"] == key), None
        )
        if entry is None:
            self._send_html("not found", status=404)
            return
        data = self.state.crop(score_id, entry)
        if data is None:
            self._send_html("scan page not found", status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--matches", type=Path, required=True,
        help="ocr_first_text_ground_truth.py's --out dir (per-score json files).",
    )
    parser.add_argument(
        "--judgments", type=Path, required=True,
        help="Where good/bad/unclear judgments are saved (created if missing).",
    )
    parser.add_argument(
        "--pngs", type=Path, required=True, nargs="+",
        help="One or more imslp_pngs dirs (both imslp_pngs and imslp_pngs_new, same "
        "split as extract_stage2_pairs.py - resolved per score by whichever dir "
        "actually has that score's own subdirectory).",
    )
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()

    Handler.state = ReviewState(args.matches, args.judgments, args.pngs)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(
        f"reviewing {len(Handler.state.entries)} match(es) across "
        f"{len(Handler.state.by_score)} score(s) at http://localhost:{args.port}/"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
