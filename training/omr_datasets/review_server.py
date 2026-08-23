"""A local review tool for `detect_imslp_systems.py`'s automated system boxes.

Serves one page: a list of scores, each opening into a per-page box editor (canvas,
drag corners to resize, drag body to move, click empty space to draw a new box,
Delete to remove the selected one). "Save & next" persists that page's boxes to a
separate `--verified` directory in the same `imslp_systems/*.yaml` schema the rest of
this project's tooling already reads (`olimpic_repair.py`, `box_probe.py`) - the
input `--systems`/`--repaired` directories are never written to, so a review session
can always be re-run from the same automated detections.

Stdlib only, deliberately: this is a short-lived personal review tool, not a service,
so a framework dependency would outweigh what it saves.
"""

# flake8: noqa: T201

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import yaml

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_score_id(raw: str) -> str | None:
    score_id = unquote(raw)
    return score_id if _ID_RE.match(score_id) else None


class ReviewState:
    """Read-only view of the automated detections plus whatever this session (or an
    earlier one) has already verified - recomputed per request rather than cached, so
    edits made through the API are reflected immediately without a restart."""

    def __init__(self, pngs_dir: Path, systems_dir: Path, verified_dir: Path) -> None:
        self.pngs_dir = pngs_dir
        self.systems_dir = systems_dir
        self.verified_dir = verified_dir
        self.verified_dir.mkdir(parents=True, exist_ok=True)

    def score_ids(self) -> list[str]:
        return sorted(p.stem for p in self.systems_dir.glob("*.yaml"))

    def detected(self, score_id: str) -> dict:
        return _load_yaml(self.systems_dir / f"{score_id}.yaml")

    def verified(self, score_id: str) -> dict:
        return _load_yaml(self.verified_dir / f"{score_id}.yaml")

    def save_page(self, score_id: str, page_number: str, systems: list, status: str) -> None:
        document = self.verified(score_id)
        pages = document.setdefault("pages", {})
        detected_page = self.detected(score_id).get("pages", {}).get(int(page_number), {})
        page_entry = pages.setdefault(int(page_number), {})
        page_entry["image"] = detected_page.get("image", page_entry.get("image"))
        page_entry["width"] = detected_page.get("width", page_entry.get("width"))
        page_entry["height"] = detected_page.get("height", page_entry.get("height"))
        page_entry["systems"] = [{"boundingBox": box} for box in systems]
        page_entry["status"] = status
        (self.verified_dir / f"{score_id}.yaml").write_text(
            yaml.safe_dump(document), encoding="utf-8"
        )

    def progress(self, score_id: str) -> tuple[int, int]:
        detected_pages = self.detected(score_id).get("pages", {})
        verified_pages = self.verified(score_id).get("pages", {})
        confirmed = sum(
            1 for p in verified_pages.values() if p.get("status") in ("confirmed", "edited")
        )
        return confirmed, len(detected_pages)


INDEX_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>IMSLP system box review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #faf9f6; color: #222; }}
h1 {{ font-size: 1.3rem; }}
table {{ border-collapse: collapse; width: 100%; max-width: 640px; }}
td, th {{ text-align: left; padding: 0.35rem 0.8rem; border-bottom: 1px solid #ddd; }}
a {{ color: #2456a8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.done {{ color: #2a7a2a; }}
.partial {{ color: #a8720f; }}
.none {{ color: #999; }}
</style></head>
<body>
<h1>IMSLP automated system-box review</h1>
<table>
<tr><th>Score</th><th>Progress</th></tr>
{rows}
</table>
</body></html>
"""

REVIEW_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Review {score_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #1e1e1e; color: #eee; }}
#toolbar {{ margin-bottom: 0.6rem; display: flex; gap: 0.8rem; align-items: center; flex-wrap: wrap; }}
button {{ font: inherit; padding: 0.35rem 0.9rem; border-radius: 5px; border: 1px solid #555;
  background: #333; color: #eee; cursor: pointer; }}
button:hover {{ background: #444; }}
button.primary {{ background: #2456a8; border-color: #2456a8; }}
#canvas-wrap {{ position: relative; display: inline-block; }}
canvas {{ border: 1px solid #555; cursor: crosshair; }}
#status {{ opacity: 0.8; }}
a {{ color: #9cc0ff; }}
</style></head>
<body>
<p><a href="/">&larr; all scores</a></p>
<div id="toolbar">
  <button id="prev">&larr; prev page</button>
  <span id="pageLabel"></span>
  <button id="next">next page &rarr;</button>
  <button id="del">delete selected box</button>
  <button id="reset">reset to detected</button>
  <button class="primary" id="save">save &amp; next</button>
  <span id="status"></span>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<p style="opacity:0.7; max-width:640px">
Drag a box's body to move it, a corner to resize it. Drag on empty space to draw a
new box. Click a box to select it (yellow), then "delete selected box" to remove it.
"Save &amp; next" writes this page's boxes and advances.
</p>
<script>
const scoreId = {score_id_json};
const pages = {pages_json};
let pageIndex = 0;
let boxes = [];
let selected = null;
let drag = null; // {{mode: 'move'|'resize-tl'|... |'new', boxIndex, startX, startY, orig}}
const HANDLE = 8;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const img = new Image();

function currentPage() {{ return pages[pageIndex]; }}

function loadPage() {{
  const p = currentPage();
  document.getElementById('pageLabel').textContent =
    'page ' + p.number + '  (' + (pageIndex + 1) + '/' + pages.length + ')  [' + p.status + ']';
  document.getElementById('status').textContent = '';
  boxes = p.systems.map(s => ({{...s}}));
  selected = null;
  img.onload = draw;
  img.src = '/image/' + p.image;  // p.image already carries the score_id prefix
}}

function scale() {{ return canvas.width / img.naturalWidth || 1; }}

function draw() {{
  const s = Math.min(1000 / img.naturalWidth, 1);
  canvas.width = Math.round(img.naturalWidth * s);
  canvas.height = Math.round(img.naturalHeight * s);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => {{
    ctx.strokeStyle = i === selected ? '#ffd23f' : '#3ddc84';
    ctx.lineWidth = 2;
    ctx.strokeRect(b.left * s, b.top * s, b.width * s, b.height * s);
  }});
}}

function hitTest(x, y) {{
  const s = scale();
  for (let i = boxes.length - 1; i >= 0; i--) {{
    const b = boxes[i];
    const bx = b.left * s, by = b.top * s, bw = b.width * s, bh = b.height * s;
    const corners = {{
      'resize-tl': [bx, by], 'resize-tr': [bx + bw, by],
      'resize-bl': [bx, by + bh], 'resize-br': [bx + bw, by + bh],
    }};
    for (const [mode, [cx, cy]] of Object.entries(corners)) {{
      if (Math.abs(x - cx) <= HANDLE && Math.abs(y - cy) <= HANDLE) return {{mode, i}};
    }}
    if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return {{mode: 'move', i}};
  }}
  return null;
}}

canvas.addEventListener('mousedown', e => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const hit = hitTest(x, y);
  if (hit) {{
    selected = hit.i;
    drag = {{mode: hit.mode, i: hit.i, startX: x, startY: y, orig: {{...boxes[hit.i]}}}};
  }} else {{
    selected = null;
    drag = {{mode: 'new', startX: x, startY: y}};
  }}
  draw();
}});

canvas.addEventListener('mousemove', e => {{
  if (!drag) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left, y = e.clientY - rect.top;
  const s = scale();
  const dx = (x - drag.startX) / s, dy = (y - drag.startY) / s;
  if (drag.mode === 'new') {{
    const left = Math.min(drag.startX, x) / s, top = Math.min(drag.startY, y) / s;
    const width = Math.abs(x - drag.startX) / s, height = Math.abs(y - drag.startY) / s;
    boxes[boxes.length] = boxes[boxes.length] && drag.newIndex !== undefined
      ? boxes[drag.newIndex] : null;
    if (drag.newIndex === undefined) {{ drag.newIndex = boxes.length; boxes.push({{}}); }}
    boxes[drag.newIndex] = {{left, top, width, height}};
    selected = drag.newIndex;
  }} else if (drag.mode === 'move') {{
    boxes[drag.i] = {{...drag.orig, left: drag.orig.left + dx, top: drag.orig.top + dy}};
  }} else if (drag.mode.startsWith('resize')) {{
    const o = drag.orig;
    let left = o.left, top = o.top, right = o.left + o.width, bottom = o.top + o.height;
    if (drag.mode.includes('l')) left = o.left + dx;
    if (drag.mode.includes('r')) right = o.left + o.width + dx;
    if (drag.mode.includes('t')) top = o.top + dy;
    if (drag.mode.includes('b')) bottom = o.top + o.height + dy;
    boxes[drag.i] = {{left, top, width: right - left, height: bottom - top}};
  }}
  draw();
}});

window.addEventListener('mouseup', () => {{ drag = null; }});

document.getElementById('del').onclick = () => {{
  if (selected !== null) {{ boxes.splice(selected, 1); selected = null; draw(); }}
}};
document.getElementById('reset').onclick = () => loadPage();
document.getElementById('prev').onclick = () => {{
  if (pageIndex > 0) {{ pageIndex--; loadPage(); }}
}};
document.getElementById('next').onclick = () => {{
  if (pageIndex < pages.length - 1) {{ pageIndex++; loadPage(); }}
}};

document.getElementById('save').onclick = async () => {{
  const p = currentPage();
  const normalized = boxes
    .filter(b => b.width > 2 && b.height > 2)
    .map(b => ({{
      left: Math.round(b.left), top: Math.round(b.top),
      width: Math.round(b.width), height: Math.round(b.height),
    }}));
  const res = await fetch('/api/score/' + scoreId + '/page/' + p.number, {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{systems: normalized, status: 'confirmed'}}),
  }});
  document.getElementById('status').textContent = res.ok ? 'saved.' : 'save failed.';
  p.status = 'confirmed';
  if (pageIndex < pages.length - 1) {{ pageIndex++; loadPage(); }}
}};

loadPage();
</script>
</body></html>
"""


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
        elif path.startswith("/image/"):
            self._image(path.removeprefix("/image/"))
        else:
            self._send_html("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        match = re.match(r"^/api/score/([^/]+)/page/(\d+)$", self.path)
        if not match:
            self._send_json({"error": "not found"}, status=404)
            return
        score_id = _safe_score_id(match.group(1))
        if score_id is None:
            self._send_json({"error": "bad score id"}, status=400)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.state.save_page(score_id, match.group(2), body.get("systems", []), body.get("status", "edited"))
        self._send_json({"ok": True})

    def _index(self) -> None:
        rows = []
        for score_id in self.state.score_ids():
            confirmed, total = self.state.progress(score_id)
            css = "done" if total and confirmed == total else ("partial" if confirmed else "none")
            rows.append(
                f'<tr><td><a href="/score/{score_id}">{score_id}</a></td>'
                f'<td class="{css}">{confirmed}/{total} pages</td></tr>'
            )
        self._send_html(INDEX_TEMPLATE.format(rows="\n".join(rows) or "<tr><td>none found</td></tr>"))

    def _score_page(self, raw_id: str) -> None:
        score_id = _safe_score_id(raw_id)
        if score_id is None:
            self._send_html("bad score id", status=400)
            return
        detected = self.state.detected(score_id)
        verified = self.state.verified(score_id)
        pages = []
        for number in sorted(detected.get("pages", {})):
            page = detected["pages"][number]
            verified_page = verified.get("pages", {}).get(number)
            systems = (
                [s["boundingBox"] for s in verified_page["systems"]]
                if verified_page
                else [s["boundingBox"] for s in page.get("systems", [])]
            )
            pages.append(
                {
                    "number": number,
                    "image": page["image"],
                    "systems": systems,
                    "status": verified_page.get("status", "unreviewed") if verified_page else "unreviewed",
                }
            )
        if not pages:
            self._send_html(f"no detected pages for {score_id}", status=404)
            return
        self._send_html(
            REVIEW_TEMPLATE.format(
                score_id=score_id,
                score_id_json=json.dumps(score_id),
                pages_json=json.dumps(pages),
            )
        )

    def _image(self, raw_path: str) -> None:
        rel = unquote(raw_path)
        if ".." in rel.split("/"):
            self._send_html("bad path", status=400)
            return
        full = self.state.pngs_dir / rel
        if not full.is_file():
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
    parser.add_argument("--pngs", type=Path, required=True, help="An imslp_pngs(_new) dir.")
    parser.add_argument(
        "--systems", type=Path, required=True,
        help="Detections to review - an imslp_systems(_new) or _repaired dir.",
    )
    parser.add_argument(
        "--verified", type=Path, required=True, help="Where confirmed/edited boxes are saved."
    )
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    Handler.state = ReviewState(args.pngs, args.systems, args.verified)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"reviewing {len(Handler.state.score_ids())} score(s) at http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
