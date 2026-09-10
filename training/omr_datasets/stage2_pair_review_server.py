"""A local review tool for `extract_stage2_pairs.py`'s output.

Serves one page per score: every extracted (crop, token-sequence) pair for that
score, image above a compact pitch-sequence summary above the raw token text,
each with Good/Bad/Unclear buttons - the same "validate before trusting at scale"
pass `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §7 calls for before mixing this data
into an actual training run, done systematically rather than one-off spot-checks.

Also optionally mounts `stage3_text_review_server.py`'s own review pages under
`/text` in this same process, when `--text-matches` is given - one browser tab,
one forwarded port, for reviewing both extraction efforts rather than needing a
second SSH tunnel just to reach a second review server.

Stdlib only, matching `review_server.py`'s own house style for this project's
short-lived personal review tools - a framework dependency would outweigh what it
saves here.
"""

# flake8: noqa: T201

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from training.omr_datasets.stage3_text_review_server import (
    ReviewState as Stage3ReviewState,
)
from training.omr_datasets.stage3_text_review_server import (
    crop_bytes as stage3_crop_bytes,
)
from training.omr_datasets.stage3_text_review_server import (
    render_index as stage3_render_index,
)
from training.omr_datasets.stage3_text_review_server import (
    render_score_page as stage3_render_score_page,
)
from training.transformer.training_vocabulary import read_tokens

_STEM_RE = re.compile(r"^(?P<score_id>.+)-sys(?P<system>\d+)-v(?P<voice>\d+)$")
_JUDGMENTS = ("good", "bad", "unclear")


def parse_stem(stem: str) -> tuple[str, int, int] | None:
    """`(score_id, system_position, voice_index)` from a pair's own filename stem
    (`extract_stage2_pairs.py`'s own `"{score_id}-sys{position}-v{voice_idx}"`
    naming), or `None` if the stem doesn't match that shape."""
    match = _STEM_RE.match(stem)
    if not match:
        return None
    return match["score_id"], int(match["system"]), int(match["voice"])


def load_manifest(manifest_path: Path) -> list[dict]:
    """One entry per manifest line - `extract_stage2_pairs.py`'s own
    `"image_path,tokens_path"` CSV shape - annotated with the `(score_id, system,
    voice)` parsed from its filename stem. Entries whose stem doesn't parse are
    skipped (defensive - `data_loader.py` itself doesn't need this shape, only this
    review tool's own per-score grouping does).
    """
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_path, tokens_path = line.split(",", 1)
        stem = Path(image_path).stem
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        score_id, system, voice = parsed
        entries.append(
            {
                "stem": stem,
                "image_path": image_path,
                "tokens_path": tokens_path,
                "score_id": score_id,
                "system": system,
                "voice": voice,
            }
        )
    entries.sort(key=lambda e: (e["score_id"], e["system"], e["voice"]))
    return entries


def pitch_summary(tokens_path: str) -> str:
    """A compact, human-scannable line like `F4 Bb4 | Bb4 A4 Bb4 C5 D5 | ...` - one
    token's worth of pitch (with its lift, if any) per note, `|` at each barline -
    for correlating quickly against a crop's own contour without reading the full
    six-column raw token table for every note."""
    symbols = read_tokens(tokens_path)
    parts = []
    for symbol in symbols:
        if symbol.rhythm == "barline":
            parts.append("|")
        elif symbol.rhythm.startswith("rest"):
            parts.append("rest")
        elif symbol.rhythm.startswith("note") and symbol.pitch not in ("", ".", "_"):
            lift = "#" if symbol.lift == "#" else "b" if symbol.lift == "b" else ""
            parts.append(f"{symbol.pitch}{lift}")
    return " ".join(parts)


class ReviewState:
    def __init__(
        self, manifest_path: Path, judgments_path: Path, rendered_dir: Path | None = None
    ) -> None:
        self.manifest_path = manifest_path
        self.judgments_path = judgments_path
        self.rendered_dir = rendered_dir
        self.entries = load_manifest(manifest_path)
        self.by_score: dict[str, list[dict]] = {}
        for entry in self.entries:
            self.by_score.setdefault(entry["score_id"], []).append(entry)

    def rendered_path(self, stem: str) -> Path | None:
        """`render_stage2_tokens.py`'s own output for this pair, if it exists yet -
        rendering is a separate batch step, so a pair reviewed before its own
        render has finished simply shows without one (see `_score_page`)."""
        if self.rendered_dir is None:
            return None
        candidate = self.rendered_dir / f"{stem}.png"
        return candidate if candidate.is_file() else None

    def score_ids(self) -> list[str]:
        return sorted(self.by_score)

    def judgments(self) -> dict[str, dict]:
        if not self.judgments_path.exists():
            return {}
        return json.loads(self.judgments_path.read_text(encoding="utf-8"))

    def progress(self, score_id: str) -> tuple[int, int]:
        judged = self.judgments()
        total = len(self.by_score.get(score_id, []))
        done = sum(1 for e in self.by_score.get(score_id, []) if e["stem"] in judged)
        return done, total

    def save_judgment(self, stem: str, judgment: str, note: str) -> None:
        if judgment not in _JUDGMENTS:
            raise ValueError(f"unknown judgment {judgment!r}")
        current = self.judgments()
        current[stem] = {"judgment": judgment, "note": note}
        self.judgments_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _safe_score_id(raw: str) -> str | None:
    score_id = unquote(raw)
    return score_id if re.match(r"^[A-Za-z0-9_-]+$", score_id) else None


INDEX_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Stage 2 pair review</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 4px 12px; border-bottom: 1px solid #ddd; text-align: left; }}
.done {{ color: #2a7a2a; }}
.partial {{ color: #a06a00; }}
.none {{ color: #888; }}
</style></head><body>
<h1>Stage 2 pair review</h1>
<p>{total_pairs} pairs across {total_scores} scores.{text_link}</p>
<table><tr><th>Score</th><th>Reviewed</th></tr>
{rows}
</table>
</body></html>"""

SCORE_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{score_id} - Stage 2 pair review</title>
<style>
body {{ font-family: sans-serif; margin: 1.5em; }}
.pair {{ border: 1px solid #ccc; border-radius: 6px; padding: 12px; margin-bottom: 16px; }}
.pair.good {{ border-color: #2a7a2a; background: #f3fbf3; }}
.pair.bad {{ border-color: #b02a2a; background: #fdf3f3; }}
.pair.unclear {{ border-color: #a06a00; background: #fdf9ee; }}
.pair img {{ max-width: 100%; display: block; margin-bottom: 6px; image-rendering: pixelated; }}
.rendered img {{ background: #fff; border: 1px solid #eee; }}
.no-render {{ color: #888; font-style: italic; margin: 4px 0; }}
.summary {{ font-family: monospace; white-space: pre-wrap; margin: 4px 0; }}
details {{ margin: 4px 0; }}
.tokens {{ font-family: monospace; font-size: 0.85em; white-space: pre; overflow-x: auto; }}
button {{ margin-right: 6px; padding: 4px 10px; }}
nav a {{ margin-right: 16px; }}
</style></head><body>
<nav><a href="/">&larr; all scores</a>{next_link}</nav>
<h1>{score_id}</h1>
<p>{count} pairs.</p>
{pairs}
<script>
function judge(stem, judgment) {{
  fetch('/api/judge', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{stem: stem, judgment: judgment}}),
  }}).then(() => {{
    const el = document.getElementById('pair-' + stem);
    el.className = 'pair ' + judgment;
  }});
}}
</script>
</body></html>"""

PAIR_TEMPLATE = """<div class="pair {css_class}" id="pair-{stem}">
<h3>{stem}</h3>
<img src="/image/{stem}" alt="{stem}">
<div class="rendered">{rendered}</div>
<div class="summary">{summary}</div>
<details><summary>raw tokens</summary><div class="tokens">{tokens}</div></details>
<button onclick="judge('{stem}', 'good')">Good</button>
<button onclick="judge('{stem}', 'bad')">Bad</button>
<button onclick="judge('{stem}', 'unclear')">Unclear</button>
<span>{current_judgment}</span>
</div>"""

_HAS_RENDER = '<img src="/rendered/{stem}" alt="rendered notation for {stem}">'
_NO_RENDER = '<p class="no-render">not rendered yet</p>'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Handler(BaseHTTPRequestHandler):
    state: ReviewState  # set on the class before serving
    stage3_state: Stage3ReviewState | None = None  # set on the class if --text-matches is given

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
        elif path.startswith("/image/"):
            self._image(path.removeprefix("/image/"))
        elif path.startswith("/rendered/"):
            self._rendered_image(path.removeprefix("/rendered/"))
        elif path == "/text" or path.startswith("/text/"):
            self._text_route(path)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_html("not found", status=404)

    def _text_route(self, path: str) -> None:
        if self.stage3_state is None:
            self._send_html("stage 3 review not configured (no --text-matches given)", status=404)
            return
        rest = path.removeprefix("/text")
        if rest in ("", "/"):
            self._send_html(stage3_render_index(self.stage3_state, base_path="/text"))
        elif rest.startswith("/score/"):
            score_id = _safe_score_id(rest.removeprefix("/score/"))
            page = (
                stage3_render_score_page(self.stage3_state, score_id, base_path="/text")
                if score_id else None
            )
            if page is None:
                self._send_html("unknown score id", status=404)
                return
            self._send_html(page)
        elif rest.startswith("/crop/"):
            parts = rest.removeprefix("/crop/").split("/", 1)
            score_id = _safe_score_id(parts[0]) if parts else None
            key = unquote(parts[1]) if len(parts) == 2 else None
            data = stage3_crop_bytes(self.stage3_state, score_id, key) if score_id and key else None
            if data is None:
                self._send_html("not found", status=404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_html("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/text/api/judge":
            self._save_text_judgment()
            return
        if self.path != "/api/judge":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        stem = body.get("stem", "")
        judgment = body.get("judgment", "")
        note = body.get("note", "")
        if not any(e["stem"] == stem for e in self.state.entries):
            self._send_json({"error": "unknown pair"}, status=404)
            return
        try:
            self.state.save_judgment(stem, judgment, note)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        self._send_json({"ok": True})

    def _save_text_judgment(self) -> None:
        if self.stage3_state is None:
            self._send_json({"error": "stage 3 review not configured"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        judgment_id = body.get("id", "")
        judgment = body.get("judgment", "")
        note = body.get("note", "")
        score_id, _, key = judgment_id.partition("/")
        if not any(
            e["score_id"] == score_id and e["key"] == key for e in self.stage3_state.entries
        ):
            self._send_json({"error": "unknown match"}, status=404)
            return
        try:
            self.stage3_state.save_judgment(judgment_id, judgment, note)
        except ValueError as e:
            self._send_json({"error": str(e)}, status=400)
            return
        self._send_json({"ok": True})

    def _index(self) -> None:
        judged = self.state.judgments()
        rows = []
        for score_id in self.state.score_ids():
            done, total = self.state.progress(score_id)
            css = "done" if total and done == total else ("partial" if done else "none")
            rows.append(
                f'<tr><td><a href="/score/{score_id}">{score_id}</a></td>'
                f'<td class="{css}">{done}/{total}</td></tr>'
            )
        text_link = (
            ' <a href="/text">Stage 3 text review &rarr;</a>' if self.stage3_state else ""
        )
        self._send_html(
            INDEX_TEMPLATE.format(
                rows="\n".join(rows) or "<tr><td>none found</td></tr>",
                total_pairs=len(self.state.entries),
                total_scores=len(self.state.by_score),
                text_link=text_link,
            )
        )
        del judged

    def _score_page(self, raw_id: str) -> None:
        score_id = _safe_score_id(raw_id)
        if score_id is None or score_id not in self.state.by_score:
            self._send_html("unknown score id", status=404)
            return
        judged = self.state.judgments()
        pairs_html = []
        for entry in self.state.by_score[score_id]:
            stem = entry["stem"]
            record = judged.get(stem)
            css_class = record["judgment"] if record else ""
            current_judgment = f"(marked {record['judgment']})" if record else ""
            summary = _escape(pitch_summary(entry["tokens_path"]))
            tokens_text = _escape(Path(entry["tokens_path"]).read_text(encoding="utf-8"))
            rendered = _HAS_RENDER.format(stem=stem) if self.state.rendered_path(stem) else _NO_RENDER
            pairs_html.append(
                PAIR_TEMPLATE.format(
                    stem=stem, css_class=css_class, summary=summary, tokens=tokens_text,
                    current_judgment=current_judgment, rendered=rendered,
                )
            )
        all_ids = self.state.score_ids()
        next_link = ""
        if score_id in all_ids:
            position = all_ids.index(score_id)
            if position + 1 < len(all_ids):
                next_link = f' <a href="/score/{all_ids[position + 1]}">next score &rarr;</a>'
        self._send_html(
            SCORE_TEMPLATE.format(
                score_id=score_id,
                count=len(self.state.by_score[score_id]),
                pairs="\n".join(pairs_html),
                next_link=next_link,
            )
        )

    def _image(self, raw_stem: str) -> None:
        stem = unquote(raw_stem)
        entry = next((e for e in self.state.entries if e["stem"] == stem), None)
        if entry is None:
            self._send_html("not found", status=404)
            return
        full = Path(entry["image_path"])
        if not full.is_file():
            self._send_html("not found", status=404)
            return
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _rendered_image(self, raw_stem: str) -> None:
        stem = unquote(raw_stem)
        full = self.state.rendered_path(stem)
        if full is None:
            self._send_html("not found", status=404)
            return
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="extract_stage2_pairs.py's --manifest output.",
    )
    parser.add_argument(
        "--judgments", type=Path, required=True,
        help="Where good/bad/unclear judgments are saved (created if missing).",
    )
    parser.add_argument(
        "--rendered", type=Path,
        help="render_stage2_tokens.py's --out dir - shows notation rendered from "
        "each pair's own tokens if given. Optional: a pair with no render yet just "
        "shows 'not rendered yet' instead of failing.",
    )
    parser.add_argument(
        "--text-matches", type=Path,
        help="ocr_first_text_ground_truth.py's --out dir - mounts "
        "stage3_text_review_server.py's own review pages under /text in this same "
        "process/port if given (so reviewing both efforts needs only one forwarded "
        "port). Omit to skip the /text section entirely.",
    )
    parser.add_argument(
        "--text-judgments", type=Path,
        help="Where the /text section's good/bad/unclear judgments are saved. "
        "Required if --text-matches is given.",
    )
    parser.add_argument(
        "--text-pngs", type=Path, nargs="+",
        help="One or more imslp_pngs dirs for the /text section's crops. "
        "Required if --text-matches is given.",
    )
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()

    Handler.state = ReviewState(args.manifest, args.judgments, args.rendered)
    text_note = ""
    if args.text_matches:
        Handler.stage3_state = Stage3ReviewState(
            args.text_matches, args.text_judgments, args.text_pngs
        )
        text_note = (
            f" + {len(Handler.stage3_state.entries)} text match(es) at /text"
        )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(
        f"reviewing {len(Handler.state.entries)} pair(s) across "
        f"{len(Handler.state.by_score)} score(s){text_note} at http://localhost:{args.port}/"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
