"""Draw text boxes on scanned pages, and write them where the detector's data loader looks.

The released `non-lyric-text` detector returns whole pages as one enormous `Tempo` region,
and the only box ground truth this project has is 299 string-quartet pages carrying 264
`Tempo` boxes between them - the second-scarcest class, on the one repertoire the failure
was not reported against.

This serves **the first page carrying music**, one per score, because that is where a Lied
states its tempo. Labelling whole scores would spend the effort on interior pages that carry
none: across the Lieder sources the aligned MusicXML shows one to three tempo directions per
score against hundreds of expression marks.

The first page carrying music is not page 1. **76 of 215 scores open with a title page**, and
serving `*-p001.png` blindly - which the first version of this did - handed back a blank
sheet 39% of the time, as the first labelling run found out the hard way. `--systems` points
at the build's detection output and the first page with a detected system is used instead.
Those title pages are not worthless as labels (a page with no text at all is exactly where
returning the whole page as one `Tempo` region is most wrong), but they are not what someone
should be asked to look at one at a time.

Stdlib only, matching `review_server.py` and `stage2_pair_review_server.py` - these review
tools are short-lived and personal, and a dependency on a web framework outlives them.

Output is `<out>/<score>/<score>:<page>.boxes.json` in exactly the shape
`training/ocr/detector_data.boxes_of` reads, so a finished directory can be passed to
`detector_box_eval --boxes` next to the existing OSSQ ground truth with nothing in between.
"""

# flake8: noqa: T201

import argparse
import html
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

#: What this run labels. One class by default, because a chooser you never change is a key
#: you can press by accident: the scarce class is Tempo, the page served is the one where a
#: Lied states it, and everything else on that page is a different job. `--classes` widens
#: it. Lyrics is never offered - a separate detector serves those, and mixing them is what
#: `fusion_policy` exists to prevent.
CLASSES = ("Tempo",)

#: Everything `detector_masks.CLASS_INDEX` knows that this tool may write.
AVAILABLE = ("DirectionText", "Dynamic")

PAGE = """<!doctype html><meta charset="utf-8"><title>Box labelling {index}/{total}</title>
<style>
 body{{font:14px system-ui;margin:0;background:#111;color:#eee}}
 header{{position:sticky;top:0;background:#1b1b1b;padding:8px 12px;display:flex;
   gap:14px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #333;z-index:5}}
 .k{{background:#2a2a2a;border-radius:4px;padding:2px 7px;font-family:ui-monospace}}
 #wrap{{position:relative;display:inline-block;margin:12px}}
 img{{display:block;max-width:none}}
 .box{{position:absolute;border:2px solid;pointer-events:none}}
 .lab{{position:absolute;font:11px ui-monospace;padding:0 3px;color:#000;white-space:nowrap}}
 button{{font:13px system-ui;padding:5px 11px;border-radius:5px;border:1px solid #444;
   background:#262626;color:#eee;cursor:pointer}}
 button.on{{background:#eee;color:#111}}
 a{{color:#8cf}}
</style>
<header>
 <strong>{score}</strong> <span>page {index} of {total}</span>
 {buttons}
 <span class="k">z</span> undo
 <span class="k">s</span> save
 <span class="k">n</span> next
 <span class="k">p</span> prev
 <span id="count"></span>
 <span id="state"></span>
</header>
<div id="wrap"><img id="page" src="/image?i={index}"></div>
<script>
const CLASSES = {classes};
const COLOURS = {{"Dynamic":"#6cf","Expression":"#7e7","StaffText":"#fc6","Tempo":"#f77"}};
let boxes = {existing};
let current = 0, drag = null;
const wrap = document.getElementById("wrap"), img = document.getElementById("page");

function redraw() {{
  [...wrap.querySelectorAll(".box,.lab")].forEach(e => e.remove());
  boxes.forEach(b => {{
    const d = document.createElement("div");
    d.className = "box"; d.style.borderColor = COLOURS[b.label];
    d.style.left = b.left+"px"; d.style.top = b.top+"px";
    d.style.width = (b.right-b.left)+"px"; d.style.height = (b.bottom-b.top)+"px";
    wrap.appendChild(d);
    const l = document.createElement("div");
    l.className = "lab"; l.textContent = b.label;
    l.style.background = COLOURS[b.label];
    l.style.left = b.left+"px"; l.style.top = Math.max(0,b.top-13)+"px";
    wrap.appendChild(l);
  }});
  document.getElementById("count").textContent = boxes.length + " box(es)";
  [...document.querySelectorAll("button[data-c]")].forEach((b,i) =>
    b.classList.toggle("on", i === current));
}}
function at(e) {{
  const r = img.getBoundingClientRect();
  return [Math.round(e.clientX-r.left), Math.round(e.clientY-r.top)];
}}
img.addEventListener("mousedown", e => {{ drag = at(e); e.preventDefault(); }});
window.addEventListener("mouseup", e => {{
  if (!drag) return;
  const [x,y] = at(e), [x0,y0] = drag; drag = null;
  const box = {{label: CLASSES[current], left: Math.min(x0,x), top: Math.min(y0,y),
                right: Math.max(x0,x), bottom: Math.max(y0,y)}};
  if (box.right-box.left > 4 && box.bottom-box.top > 4) {{ boxes.push(box); redraw(); }}
}});
function save(then) {{
  fetch("/save?i={index}", {{method:"POST", body: JSON.stringify(boxes)}})
    .then(r => r.text()).then(t => {{
      document.getElementById("state").textContent = "saved " + t;
      if (then) location = then;
    }});
}}
document.addEventListener("keydown", e => {{
  if (CLASSES.length > 1 && e.key >= "1" && e.key <= String(CLASSES.length)) {{
    current = +e.key-1; redraw();
  }}
  else if (e.key === "z") {{ boxes.pop(); redraw(); }}
  else if (e.key === "s") save(null);
  else if (e.key === "n") save("/?i={next}");
  else if (e.key === "p") save("/?i={prev}");
}});
[...document.querySelectorAll("button[data-c]")].forEach((b,i) =>
  b.onclick = () => {{ current = i; redraw(); }});
document.getElementById("save").onclick = () => save(null);
document.getElementById("next").onclick = () => save("/?i={next}");
redraw();
</script>
"""


def _pages(root: Path, limit: int, systems: Path | None = None) -> list[Path]:
    """One page per score: the first that detection found a system on.

    Falls back to `*-p001.png` when no detection output is supplied, which is what the
    first version did for every score.
    """
    if systems is None:
        found = sorted(root.glob("*/*-p001.png"))
        return found[:limit] if limit else found

    import yaml

    found = []
    for document_path in sorted(systems.glob("*.yaml")):
        document = yaml.safe_load(document_path.read_text(encoding="utf-8")) or {}
        named = sorted(
            Path(page["image"]).name
            for page in (document.get("pages") or {}).values()
            if page.get("systems")
        )
        if not named:
            continue
        page = root / document_path.stem / named[0]
        if page.is_file():
            found.append(page)
    return found[:limit] if limit else found


def _record_path(out: Path, page: Path) -> Path:
    """`<score>/<score>:<page>.boxes.json`, zero-padded to four as the OSSQ set is.

    `detector_data.collect` globs `*/*.boxes.json` and resolves each record's `image`
    field *relative to the record*, so the page has to sit beside it - see `_store`.
    """
    score = page.parent.name
    number = re.sub(r"^.*-p(\d+)\.png$", r"\1", page.name)
    return out / score / f"{score}:{int(number):04d}.boxes.json"


def _load(out: Path, page: Path) -> list[dict]:
    record = _record_path(out, page)
    if not record.is_file():
        return []
    payload = json.loads(record.read_text(encoding="utf-8"))
    return [
        {"label": label, **box}
        for label, group in payload.get("text_boxes", {}).items()
        for box in group
    ]


def _store(out: Path, page: Path, boxes: list[dict]) -> Path:
    grouped: dict[str, list[dict]] = {}
    for box in boxes:
        label = box.get("label")
        if label not in AVAILABLE:
            continue
        grouped.setdefault(label, []).append(
            {key: int(box[key]) for key in ("left", "top", "right", "bottom")}
        )
    record = _record_path(out, page)
    record.parent.mkdir(parents=True, exist_ok=True)
    # The reader resolves `image` against the record's own directory, so the page must be
    # reachable from there. Symlinked rather than copied: these are 1 MB scans and the
    # build directory they come from is the one that gets rebuilt, not this one.
    beside = record.with_suffix("").with_suffix(".png")
    if not beside.exists():
        try:
            beside.symlink_to(page.resolve())
        except OSError:
            beside.write_bytes(page.read_bytes())
    # `lyrics` is written empty rather than omitted so the file shape never varies -
    # `detector_data.boxes_of` reads that key directly. Same rule as ossq_box_ground_truth.
    record.write_text(
        json.dumps({"image": beside.name, "lyrics": [], "text_boxes": grouped}),
        encoding="utf-8",
    )
    return record


def serve(pages: list[Path], out: Path, port: int, classes: tuple[str, ...] = CLASSES) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _index(self) -> int:
            query = parse_qs(urlparse(self.path).query)
            return max(1, min(len(pages), int(query.get("i", ["1"])[0])))

        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            index = self._index()
            page = pages[index - 1]
            if route == "/image":
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            # With one class there is nothing to choose, so nothing is offered.
            buttons = (
                ""
                if len(classes) == 1
                else " ".join(
                    f'<button data-c="{i}">{i + 1} {html.escape(name)}</button>'
                    for i, name in enumerate(classes)
                )
            )
            buttons += ' <button id="save">save</button> <button id="next">save + next</button>'
            body = PAGE.format(
                index=index,
                total=len(pages),
                score=html.escape(page.parent.name),
                classes=json.dumps(classes),
                existing=json.dumps(_load(out, page)),
                buttons=buttons,
                next=min(len(pages), index + 1),
                prev=max(1, index - 1),
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            index = self._index()
            length = int(self.headers.get("Content-Length", "0"))
            boxes = json.loads(self.rfile.read(length) or b"[]")
            record = _store(out, pages[index - 1], boxes)
            body = f"{len(boxes)} box(es) -> {record.name}".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"{len(pages)} pages, labelling {', '.join(classes)}, writing to {out}")
    print(f"open http://localhost:{port}/   (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pages", type=Path, required=True, help="a _build/pages directory")
    parser.add_argument("--out", type=Path, required=True, help="where .boxes.json is written")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument(
        "--classes",
        default=",".join(CLASSES),
        help=f"comma-separated, from {', '.join(AVAILABLE)}",
    )
    parser.add_argument(
        "--systems",
        type=Path,
        help="a _build/systems directory; serves the first page with music instead of p001",
    )
    args = parser.parse_args()
    pages = _pages(args.pages, args.limit, args.systems)
    if not pages:
        raise SystemExit(f"no *-p001.png under {args.pages}")
    classes = tuple(name.strip() for name in args.classes.split(",") if name.strip())
    unknown = [name for name in classes if name not in AVAILABLE]
    if unknown:
        raise SystemExit(f"not detector classes: {', '.join(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    serve(pages, args.out, args.port, classes)


if __name__ == "__main__":
    main()
