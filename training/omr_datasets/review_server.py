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

    def __init__(
        self,
        pngs_dir: Path,
        systems_dir: Path,
        verified_dir: Path,
        targeted_candidates_path: Path | None = None,
        ground_truth_renders_dir: Path | None = None,
        match_review_path: Path | None = None,
    ) -> None:
        self.pngs_dir = pngs_dir
        self.systems_dir = systems_dir
        self.verified_dir = verified_dir
        self.verified_dir.mkdir(parents=True, exist_ok=True)
        self.targeted_candidates_path = targeted_candidates_path
        self.ground_truth_renders_dir = ground_truth_renders_dir
        self.match_review_path = match_review_path

    def targeted_candidates(self) -> list[dict]:
        """`targeted_review_candidates.py`'s own output - specific pages worth a
        second look (a bar-count mismatch on a score whose other pages otherwise
        agree with its matched Lieder ground truth), not every page of every score.
        """
        if not self.targeted_candidates_path or not self.targeted_candidates_path.exists():
            return []
        return json.loads(self.targeted_candidates_path.read_text(encoding="utf-8"))

    def ground_truth_pages(self, score_id: str) -> list[str]:
        """Rendered ground-truth page filenames (`render_lieder_ground_truth.py`'s
        own output) for one score, in page order - the *whole* piece, not just the
        one flagged page, so a human can tell whether the flagged scan page exists
        anywhere in the matched transcription at all, not just whether that one
        page matches."""
        if not self.ground_truth_renders_dir:
            return []
        score_dir = self.ground_truth_renders_dir / score_id
        if not score_dir.exists():
            return []
        return sorted(p.name for p in score_dir.glob("page*.png"))

    def compare_items(self) -> list[dict]:
        """One item per unique (score, flagged page) - several flagged systems on
        the same page share the same "does this page even match" question, so
        they're grouped rather than reviewed one at a time."""
        by_key: dict[tuple[str, int], dict] = {}
        for c in self.targeted_candidates():
            key = (c["score_id"], c["page_index"])
            item = by_key.setdefault(
                key,
                {
                    "score_id": c["score_id"],
                    "page_index": c["page_index"],
                    "page_image": c["page_image"],
                    "is_first_page": c["is_first_page"],
                    "is_last_page": c["is_last_page"],
                    "mismatches": [],
                },
            )
            item["mismatches"].append(
                {
                    "system_index": c["system_index"],
                    "detected": c["detected"],
                    "ground_truth": c["ground_truth"],
                }
            )
        items = sorted(by_key.values(), key=lambda item: (item["score_id"], item["page_index"]))
        judgments = self.match_judgments()
        for item in items:
            saved = judgments.get(f"{item['score_id']}/{item['page_index']}")
            item["judgment"] = saved["judgment"] if saved else None
            item["confirmed_gt_page"] = saved["confirmed_gt_page"] if saved else None
        return items

    def match_judgments(self) -> dict:
        if not self.match_review_path or not self.match_review_path.exists():
            return {}
        return json.loads(self.match_review_path.read_text(encoding="utf-8"))

    def save_match_judgment(
        self, score_id: str, page_index: int, judgment: str, confirmed_gt_page: int, note: str
    ) -> None:
        if not self.match_review_path:
            return
        data = self.match_judgments()
        data[f"{score_id}/{page_index}"] = {
            "score_id": score_id,
            "page_index": page_index,
            "judgment": judgment,
            "confirmed_gt_page": confirmed_gt_page,
            "note": note,
        }
        self.match_review_path.parent.mkdir(parents=True, exist_ok=True)
        self.match_review_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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
<p><a href="/targeted">targeted review candidates</a> - specific pages a bar-count
check flagged, instead of browsing every score below.</p>
<table>
<tr><th>Score</th><th>Progress</th></tr>
{rows}
</table>
</body></html>
"""

TARGETED_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Targeted review candidates</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #faf9f6; color: #222; }}
h1 {{ font-size: 1.3rem; }}
table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
td, th {{ text-align: left; padding: 0.35rem 0.8rem; border-bottom: 1px solid #ddd; }}
a {{ color: #2456a8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.edge {{ color: #a8720f; font-weight: 600; }}
</style></head>
<body>
<h1>Targeted review candidates ({count})</h1>
<p><a href="/">&larr; all scores</a> · <a href="/compare">side-by-side comparison view</a></p>
<p style="opacity:0.7; max-width:640px">
Pages where a bar-count check against the score's matched Lieder transcription
disagreed, on a score whose *other* pages otherwise agree - a specific, targeted
signal that this page (not the whole score) is worth a second look, rather than a
blanket re-check of every low-scoring score. The side-by-side view is the faster
way to go through these one at a time.
</p>
<table>
<tr><th>Score / page</th><th>Detected vs. ground truth</th><th>Position</th><th></th></tr>
{rows}
</table>
</body></html>
"""

GROUND_TRUTH_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Ground truth: {score_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #faf9f6; color: #222; }}
h1 {{ font-size: 1.3rem; }}
a {{ color: #2456a8; }}
img {{ display: block; max-width: 100%; margin: 0 0 1.5rem 0; border: 1px solid #ccc; }}
</style></head>
<body>
<h1>Ground truth render: {score_id}</h1>
<p><a href="/targeted">&larr; targeted review candidates</a> ·
<a href="/score/{score_id}">the scanned score</a></p>
<p style="opacity:0.7; max-width:640px">
Rendered from the matched Lieder transcription's own MuseScore source - the whole
piece, every page, so you can tell whether the flagged scan page exists anywhere in
here at all, not just whether that one page matches.
</p>
{images}
</body></html>
"""

COMPARE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Compare scan vs. ground truth</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; background: #1e1e1e; color: #eee; }}
#toolbar {{ margin-bottom: 0.6rem; display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }}
button {{ font: inherit; padding: 0.35rem 0.9rem; border-radius: 5px; border: 1px solid #555;
  background: #333; color: #eee; cursor: pointer; }}
button:hover {{ background: #444; }}
button.match {{ background: #2a7a2a; border-color: #2a7a2a; }}
button.diff-layout {{ background: #2456a8; border-color: #2456a8; }}
button.no-match {{ background: #a8720f; border-color: #a8720f; }}
a {{ color: #9cc0ff; }}
#panes {{ display: flex; gap: 1rem; align-items: flex-start; }}
.pane {{ flex: 1; min-width: 0; }}
.pane img {{ max-width: 100%; border: 1px solid #555; }}
.pane h3 {{ font-size: 0.9rem; margin: 0 0 0.4rem 0; opacity: 0.85; }}
#status {{ opacity: 0.85; }}
#noteInput {{ padding: 0.35rem; }}
table.mismatches {{ border-collapse: collapse; margin: 0.6rem 0; font-size: 0.85rem; }}
table.mismatches td {{ padding: 0.15rem 0.7rem; border-bottom: 1px solid #444; }}
.judged-match {{ color: #6cc06c; }}
.judged-different_layout {{ color: #7aa8e0; }}
.judged-no_match {{ color: #e0a94a; }}
</style></head>
<body>
<p><a href="/targeted">&larr; targeted review candidates</a> ({count} page(s) to compare)</p>
<div id="toolbar">
  <button id="prev">&larr; prev</button>
  <span id="itemLabel"></span>
  <button id="next">next &rarr;</button>
  <span style="flex:1"></span>
  <button id="gtPrev">ground truth page &larr;</button>
  <span id="gtPageLabel"></span>
  <button id="gtNext">ground truth page &rarr;</button>
</div>
<div id="panes">
  <div class="pane"><h3 id="scanLabel"></h3><img id="scanImg"></div>
  <div class="pane"><h3 id="gtLabel"></h3><img id="gtImg"></div>
</div>
<table class="mismatches" id="mismatchTable"></table>
<div id="toolbar">
  <button class="match" id="matchBtn">match</button>
  <button class="diff-layout" id="diffLayoutBtn"
    title="Same piece, same starting bar, but the transcription's own line breaks
don't correspond to the scan's - per-system bar counts aren't comparable here, but
the source/page match itself is still confirmed correct.">different layout</button>
  <button class="no-match" id="noMatchBtn">no match</button>
  <input id="noteInput" placeholder="optional note" style="flex:1">
  <span id="status"></span>
</div>
<script>
const items = {items_json};
let index = 0;
let gtPageOffset = 0;

function currentItem() {{ return items[index]; }}

function render() {{
  const item = currentItem();
  const judgedClass = item.judgment ? ' judged-' + item.judgment : '';
  document.getElementById('itemLabel').innerHTML =
    (index + 1) + '/' + items.length + '  ' + item.score_id +
    '  <span class="' + judgedClass + '">' + (item.judgment || 'unreviewed') + '</span>';
  document.getElementById('scanLabel').textContent = 'scan: ' + item.page_image;
  document.getElementById('scanImg').src = '/image/' + item.page_image;

  gtPageOffset = item.confirmed_gt_page ? item.confirmed_gt_page - 1 - item.page_index : 0;
  updateGtImage();

  document.getElementById('mismatchTable').innerHTML = item.mismatches.map(m =>
    '<tr><td>system ' + m.system_index + '</td><td>detected ' + m.detected +
    '</td><td>ground truth ' + m.ground_truth + '</td></tr>'
  ).join('');
  document.getElementById('status').textContent = '';
  document.getElementById('noteInput').value = '';
}}

function updateGtImage() {{
  const item = currentItem();
  const gtIndex = item.page_index + gtPageOffset;
  const pageNum = String(Math.max(0, gtIndex) + 1).padStart(3, '0');
  document.getElementById('gtImg').src = '/ground_truth_image/' + item.score_id + '/page' + pageNum + '.png';
  document.getElementById('gtLabel').textContent = 'ground truth page ' + (gtIndex + 1) +
    (item.is_first_page ? ' (scan: first page)' : item.is_last_page ? ' (scan: last page)' : '');
  document.getElementById('gtPageLabel').textContent = 'page ' + (gtIndex + 1);
}}

document.getElementById('prev').onclick = () => {{ if (index > 0) {{ index--; render(); }} }};
document.getElementById('next').onclick = () => {{ if (index < items.length - 1) {{ index++; render(); }} }};
document.getElementById('gtPrev').onclick = () => {{ gtPageOffset--; updateGtImage(); }};
document.getElementById('gtNext').onclick = () => {{ gtPageOffset++; updateGtImage(); }};

async function submitJudgment(judgment) {{
  const item = currentItem();
  const gtIndex = item.page_index + gtPageOffset;
  const res = await fetch('/api/compare', {{
    method: 'POST', headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{
      score_id: item.score_id, page_index: item.page_index, judgment,
      confirmed_gt_page: gtIndex + 1,
      note: document.getElementById('noteInput').value,
    }}),
  }});
  if (res.ok) {{
    item.judgment = judgment;
    item.confirmed_gt_page = gtIndex + 1;
    document.getElementById('status').textContent = 'saved.';
    if (index < items.length - 1) {{ index++; render(); }}
  }} else {{
    document.getElementById('status').textContent = 'save failed.';
  }}
}}

document.getElementById('matchBtn').onclick = () => submitJudgment('match');
document.getElementById('diffLayoutBtn').onclick = () => submitJudgment('different_layout');
document.getElementById('noMatchBtn').onclick = () => submitJudgment('no_match');

render();
</script>
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
.spinner {{ display: inline-block; width: 1em; height: 1em; border-radius: 50%;
  border: 2px solid #666; border-top-color: #eee; vertical-align: middle;
  animation: spin 0.7s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style></head>
<body>
<p><a href="/">&larr; all scores</a></p>
<div id="toolbar">
  <button id="prev">&larr; prev page</button>
  <span id="pageLabel"></span>
  <button id="next">next page &rarr;</button>
  <button id="del">delete selected box</button>
  <button id="reset">reset to detected</button>
  <button class="primary" id="save">save this page &amp; next</button>
  <span id="spinner" class="spinner" style="display:none"></span>
  <span id="status"></span>
</div>
<div id="canvas-wrap"><canvas id="c"></canvas></div>
<p style="opacity:0.7; max-width:640px">
Drag a box's body to move it, a corner to resize it. Drag on empty space to draw a
new box. Click a box to select it (yellow), then "delete selected box" to remove it.
"Save this page &amp; next" saves only the page you're looking at, then advances -
prev/next page never saves for you, so a page you edit but don't explicitly save is
left as detected. On the last page of a score it moves on to the next score in the
list instead.
</p>
<script>
const scoreId = {score_id_json};
const nextScoreId = {next_score_id_json};
const scorePosition = {score_position_json};
const scoreTotal = {score_total_json};
const pages = {pages_json};
let pageIndex = 0;
{{
  // A /targeted link opens straight to the flagged page (?page=<page number>),
  // not always page 1 - find it by page.number, not array index, since a score's
  // own page numbering can start above 1 (cover pages the detector already skips).
  const requestedPage = new URLSearchParams(window.location.search).get('page');
  if (requestedPage !== null) {{
    const found = pages.findIndex(p => String(p.number) === requestedPage);
    if (found >= 0) pageIndex = found;
  }}
}}
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
    'score ' + scorePosition + '/' + scoreTotal + '  page ' + p.number + '  (' +
    (pageIndex + 1) + '/' + pages.length + ')  [' + p.status + ']';
  document.getElementById('status').textContent = '';
  boxes = p.systems.map(s => ({{...s}}));
  selected = null;
  img.onload = draw;
  img.src = '/image/' + p.image;  // p.image already carries the score_id prefix
}}

function scale() {{ return canvas.width / img.naturalWidth || 1; }}

function boxColor(i) {{
  // Golden-angle hue step - stays visually distinct box to box even for a page with
  // many systems, unlike an even N-way split which repeats/clusters as N grows.
  const hue = (i * 137.508) % 360;
  return `hsl(${{hue}}, 75%, 60%)`;
}}

function draw() {{
  const s = Math.min(1000 / img.naturalWidth, 1);
  canvas.width = Math.round(img.naturalWidth * s);
  canvas.height = Math.round(img.naturalHeight * s);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => {{
    ctx.strokeStyle = i === selected ? '#ffd23f' : boxColor(i);
    ctx.lineWidth = i === selected ? 3 : 2;
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
    if (drag.newIndex === undefined) {{
      drag.newIndex = boxes.length;
      boxes.push({{left: 0, top: 0, width: 0, height: 0}});
    }}
    const left = Math.min(drag.startX, x) / s, top = Math.min(drag.startY, y) / s;
    const width = Math.abs(x - drag.startX) / s, height = Math.abs(y - drag.startY) / s;
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
  const saveButton = document.getElementById('save');
  const spinner = document.getElementById('spinner');
  const p = currentPage();
  const normalized = boxes
    .filter(b => b.width > 2 && b.height > 2)
    .map(b => ({{
      left: Math.round(b.left), top: Math.round(b.top),
      width: Math.round(b.width), height: Math.round(b.height),
    }}));
  saveButton.disabled = true;
  spinner.style.display = 'inline-block';
  document.getElementById('status').textContent = 'saving...';
  let res;
  try {{
    res = await fetch('/api/score/' + scoreId + '/page/' + p.number, {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{systems: normalized, status: 'confirmed'}}),
    }});
  }} catch (e) {{
    spinner.style.display = 'none';
    saveButton.disabled = false;
    document.getElementById('status').textContent = 'save failed: ' + e;
    return;
  }}
  if (!res.ok) {{
    spinner.style.display = 'none';
    saveButton.disabled = false;
    document.getElementById('status').textContent = 'save failed.';
    return;
  }}
  p.status = 'confirmed';
  p.systems = normalized;  // keep in-memory state in sync - prev/next re-reads this
  // The spinner stays visible through the page/score transition below - loadPage's
  // own image fetch (or the navigation itself) is the rest of what "saving" is
  // waiting on from the user's point of view, not just the POST above.
  if (pageIndex < pages.length - 1) {{
    pageIndex++;
    loadPage();
    img.addEventListener('load', () => {{
      spinner.style.display = 'none';
      saveButton.disabled = false;
    }}, {{once: true}});
  }} else if (nextScoreId) {{
    document.getElementById('status').textContent = 'saved. moving to next score...';
    setTimeout(() => {{ window.location.href = '/score/' + nextScoreId; }}, 500);
  }} else {{
    document.getElementById('status').textContent = 'saved. last score - back to list...';
    setTimeout(() => {{ window.location.href = '/'; }}, 700);
  }}
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
        elif path == "/targeted":
            self._targeted()
        elif path == "/compare":
            self._compare()
        elif path.startswith("/score/"):
            self._score_page(path.removeprefix("/score/"))
        elif path.startswith("/image/"):
            self._image(path.removeprefix("/image/"))
        elif path.startswith("/ground_truth_image/"):
            self._ground_truth_image(path.removeprefix("/ground_truth_image/"))
        elif path.startswith("/ground_truth/"):
            self._ground_truth_page(path.removeprefix("/ground_truth/"))
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send_html("not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/compare":
            self._save_compare_judgment()
            return
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

    def _save_compare_judgment(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        score_id = _safe_score_id(body.get("score_id", ""))
        if score_id is None:
            self._send_json({"error": "bad score id"}, status=400)
            return
        self.state.save_match_judgment(
            score_id,
            int(body.get("page_index", -1)),
            body.get("judgment", ""),
            int(body.get("confirmed_gt_page", 0)),
            body.get("note", ""),
        )
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

    def _targeted(self) -> None:
        candidates = self.state.targeted_candidates()
        candidates.sort(key=lambda c: (c["score_id"], c["page_index"], c["system_index"]))
        rows = []
        for c in candidates:
            match = re.search(r"-p(\d+)\.png$", c["page_image"])
            page_number = int(match.group(1)) if match else None
            position = (
                "first page" if c["is_first_page"]
                else "last page" if c["is_last_page"]
                else ""
            )
            css = ' class="edge"' if position else ""
            has_render = bool(self.state.ground_truth_pages(c["score_id"]))
            ground_truth_link = (
                f'<a href="/ground_truth/{c["score_id"]}">ground truth</a>' if has_render else ""
            )
            rows.append(
                f'<tr><td><a href="/score/{c["score_id"]}?page={page_number}">'
                f'{c["score_id"]} p{page_number} (system {c["system_index"]})</a></td>'
                f'<td>{c["detected"]} vs {c["ground_truth"]}</td>'
                f"<td{css}>{position}</td>"
                f"<td>{ground_truth_link}</td></tr>"
            )
        self._send_html(
            TARGETED_TEMPLATE.format(
                count=len(candidates),
                rows="\n".join(rows) or "<tr><td>none found</td></tr>",
            )
        )

    def _compare(self) -> None:
        items = self.state.compare_items()
        self._send_html(COMPARE_TEMPLATE.format(count=len(items), items_json=json.dumps(items)))

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
        all_ids = self.state.score_ids()
        next_score_id = None
        score_position = None
        if score_id in all_ids:
            position = all_ids.index(score_id)
            score_position = position + 1
            if position + 1 < len(all_ids):
                next_score_id = all_ids[position + 1]
        self._send_html(
            REVIEW_TEMPLATE.format(
                score_id=score_id,
                score_id_json=json.dumps(score_id),
                pages_json=json.dumps(pages),
                next_score_id_json=json.dumps(next_score_id),
                score_position_json=json.dumps(score_position),
                score_total_json=json.dumps(len(all_ids)),
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

    def _ground_truth_page(self, raw_id: str) -> None:
        score_id = _safe_score_id(raw_id)
        if score_id is None:
            self._send_html("bad score id", status=400)
            return
        page_names = self.state.ground_truth_pages(score_id)
        if not page_names:
            self._send_html(
                f"no ground-truth render for {score_id} - run render_lieder_ground_truth.py "
                "for it first",
                status=404,
            )
            return
        images = "\n".join(
            f'<img src="/ground_truth_image/{score_id}/{name}" alt="{name}">'
            for name in page_names
        )
        self._send_html(GROUND_TRUTH_TEMPLATE.format(score_id=score_id, images=images))

    def _ground_truth_image(self, raw_path: str) -> None:
        rel = unquote(raw_path)
        if ".." in rel.split("/") or not self.state.ground_truth_renders_dir:
            self._send_html("bad path", status=400)
            return
        full = self.state.ground_truth_renders_dir / rel
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
    parser.add_argument(
        "--targeted-candidates", type=Path,
        help="targeted_review_candidates.py's --out file - powers the /targeted page.",
    )
    parser.add_argument(
        "--ground-truth-renders", type=Path,
        help="render_lieder_ground_truth.py's --out dir - powers the /ground_truth/<id> page.",
    )
    parser.add_argument(
        "--match-review", type=Path,
        help="Where /compare's match/no-match judgments are saved.",
    )
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()

    Handler.state = ReviewState(
        args.pngs, args.systems, args.verified,
        args.targeted_candidates, args.ground_truth_renders, args.match_review,
    )
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"reviewing {len(Handler.state.score_ids())} score(s) at http://localhost:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
