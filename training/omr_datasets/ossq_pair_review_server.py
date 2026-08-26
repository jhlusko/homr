"""Review whether each OSSQ scanned staff crop matches its own label.

After the pagination fix (§7), the question worth a reviewer's time changed, and this
page changed with it.

**The comparison this page originally made is no longer meaningful.** It showed the
scanned crop beside the synthetic crop of the same staff name, on the premise that the
same name denotes the same music in both tracks. That premise held only while
`convert_ossq.py` took *both* tracks' symbols from `musicxml/unaligned` - which is
precisely the bug that mislabelled 56.7% of the scanned corpus. Measured on the first 16
reviewed staves: the synthetic and scanned labels were byte-identical 16/16 before the
fix, and 8/16 after it. Each track is now keyed to its own pagination, so one filename
means page 14 of a 24-page render *and* page 14 of a 22-page scan - different music
wherever the layouts diverge, which is expected rather than wrong.

So the crops differing is not evidence of a fault, and a reviewer asked to judge it will
report faults that are not there. The question that *does* decide correctness is whether
a scanned crop matches **its own** label.

That question is asked visually. The label is rendered back into notation through
`render_stage2_tokens.py`, so the page shows two pictures of music - the scan, and what
its label says the scan contains - and the reviewer compares those. Pitch token
sequences are also on the page, but underneath and as a fallback: reading
`D4 F3 F3 F3 G3 ...` against a photograph is a much harder thing to ask of a person than
comparing two staves, and a review tool that is unpleasant to use gets used less and
less carefully.
"""

# flake8: noqa: T201

import argparse
import html
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

#: Below this scanned accuracy a staff is treated as collapsed rather than degraded -
#: the same 50-point-drop threshold `domain_gap.py` reports on, expressed absolutely.
COLLAPSE_BELOW = 0.5


def read_index(path: Path) -> dict[str, tuple[str, str]]:
    """`{staff name: (image, tokens)}` - keyed by the token filename, which is what the
    two tracks share and what makes them comparable at all."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, tokens = line.split(",")
        found[Path(tokens).name] = (image, tokens)
    return found


def read_predictions(path: Path, field: str = "pitch") -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not path or not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        reference = record.get(f"{field}_reference") or []
        predicted = record.get(f"{field}_predicted") or []
        if not reference:
            continue
        matched = sum(1 for want, got in zip(reference, predicted) if want == got)
        found[Path(record["tokens"]).name] = {
            "reference": reference,
            "predicted": predicted,
            "accuracy": matched / len(reference),
        }
    return found


class ReviewState:
    def __init__(
        self,
        synthetic_index: Path,
        scanned_index: Path,
        judgments_path: Path,
        synthetic_predictions: Path | None = None,
        scanned_predictions: Path | None = None,
        rendered_dir: Path | None = None,
    ) -> None:
        self.synthetic = read_index(synthetic_index)
        self.scanned = read_index(scanned_index)
        self.judgments_path = judgments_path
        self.rendered_dir = rendered_dir
        self.synthetic_predictions = read_predictions(synthetic_predictions) if synthetic_predictions else {}
        self.scanned_predictions = read_predictions(scanned_predictions) if scanned_predictions else {}

    def staves(self) -> list[str]:
        """Shared staves, worst scanned accuracy first.

        Staves with no prediction sort last rather than first: an absent measurement is
        not evidence of a collapse, and putting them at the top would fill the page
        with rows that say nothing.
        """
        shared = sorted(set(self.synthetic) & set(self.scanned))
        return sorted(
            shared,
            key=lambda name: self.scanned_predictions.get(name, {}).get("accuracy", 2.0),
        )

    def judgments(self) -> dict[str, dict]:
        if not self.judgments_path.is_file():
            return {}
        return json.loads(self.judgments_path.read_text(encoding="utf-8"))

    def next_unjudged(self, current: str) -> str | None:
        """The next staff after `current` that has no verdict yet.

        Reviewing is a queue, so recording a verdict should hand back the next thing to
        look at rather than the page just finished. Skipping already-judged staves is
        what makes that true on a second pass: returning to a reviewed run would
        otherwise walk through decisions already made.

        Wraps to the start so a reviewer who began in the middle still reaches the
        staves before their entry point, and returns None only when nothing is left.
        """
        staves = self.staves()
        judged = self.judgments()
        if current in staves:
            order = staves[staves.index(current) + 1 :] + staves[: staves.index(current)]
        else:
            order = staves
        return next((name for name in order if name not in judged), None)

    def save_judgment(self, staff: str, judgment: str, note: str) -> None:
        existing = self.judgments()
        existing[staff] = {"judgment": judgment, "note": note}
        self.judgments_path.write_text(json.dumps(existing, indent=1), encoding="utf-8")

    def image_path(self, track: str, staff: str) -> Path | None:
        if track == "rendered":
            if self.rendered_dir is None:
                return None
            path = self.rendered_dir / f"{Path(staff).stem}.png"
            return path if path.is_file() else None
        table = self.synthetic if track == "synthetic" else self.scanned
        entry = table.get(staff)
        if entry is None:
            return None
        path = Path(entry[0])
        return path if path.is_file() else None

    def tokens_text(self, staff: str) -> str:
        entry = self.scanned.get(staff)
        if entry is None:
            return ""
        path = Path(entry[1])
        return path.read_text(encoding="utf-8") if path.is_file() else ""


_SAFE = re.compile(r"^[A-Za-z0-9_.:-]+$")


def safe_staff(raw: str) -> str | None:
    return raw if _SAFE.match(raw or "") else None


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _sequence(values: list[str], limit: int = 60) -> str:
    shown = " ".join(values[:limit])
    return _escape(shown + (" …" if len(values) > limit else ""))


def render_index(state: ReviewState, offset: int = 0, page_size: int = 25) -> str:
    staves = state.staves()
    judged = state.judgments()
    window = staves[offset : offset + page_size]
    collapsed = sum(
        1
        for name in staves
        if state.scanned_predictions.get(name, {}).get("accuracy", 2.0) < COLLAPSE_BELOW
    )
    rows = []
    for name in window:
        scanned = state.scanned_predictions.get(name, {})
        synthetic = state.synthetic_predictions.get(name, {})
        verdict = judged.get(name, {}).get("judgment", "")
        syn_acc = synthetic.get("accuracy")
        scn_acc = scanned.get("accuracy")
        rows.append(
            f"<tr><td><a href='/staff?name={_escape(name)}'>{_escape(name)}</a></td>"
            f"<td class='n'>{'' if syn_acc is None else f'{syn_acc:.0%}'}</td>"
            f"<td class='n'>{'' if scn_acc is None else f'{scn_acc:.0%}'}</td>"
            f"<td>{_escape(verdict)}</td></tr>"
        )
    nav = []
    if offset:
        nav.append(f"<a href='/?offset={max(0, offset - page_size)}'>&larr; previous</a>")
    if offset + page_size < len(staves):
        nav.append(f"<a href='/?offset={offset + page_size}'>next &rarr;</a>")
    return f"""<!doctype html><meta charset='utf-8'><title>OSSQ pair review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:70rem}}
table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #ddd;padding:.4rem .6rem;text-align:left}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
nav a{{margin-right:1rem}}
</style>
<h1>OSSQ scanned vs synthetic crops</h1>
<p>{len(staves):,} paired staves, {collapsed:,} scoring under {COLLAPSE_BELOW:.0%} on the
scan. Worst first. {len(judged):,} judged.</p>
<table><tr><th>staff</th><th>synthetic</th><th>scanned</th><th>verdict</th></tr>
{''.join(rows)}</table>
<nav>{' '.join(nav)}</nav>"""


def render_staff(state: ReviewState, staff: str) -> str | None:
    if staff not in state.scanned or staff not in state.synthetic:
        return None
    scanned = state.scanned_predictions.get(staff, {})
    synthetic = state.synthetic_predictions.get(staff, {})
    judged = state.judgments().get(staff, {})
    reference = scanned.get("reference") or synthetic.get("reference") or []
    return f"""<!doctype html><meta charset='utf-8'><title>{_escape(staff)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:80rem}}
img{{max-width:100%;border:1px solid #ccc;background:#fff}}
h2{{margin-bottom:.2rem;font-size:1rem}}
.hint{{color:#555;font-size:.85rem;margin:.3rem 0 1rem}}
code{{display:block;white-space:pre-wrap;background:#f6f6f6;padding:.5rem;font-size:.8rem}}
.acc{{font-variant-numeric:tabular-nums}}
</style>
<p><a href='/'>&larr; all staves</a></p>
<h1>{_escape(staff)}</h1>
<p class='acc'>synthetic {synthetic.get('accuracy', float('nan')):.0%} &middot;
scanned {scanned.get('accuracy', float('nan')):.0%}</p>
<h2>the scan</h2><img src='/image?track=scanned&name={_escape(staff)}'>
<h2>what its label says the scan contains</h2><img src='/image?track=rendered&name={_escape(staff)}'>
<p class='hint'>Same music? Then the label is right. The rendering is engraved fresh, so
spacing, beaming and line breaks will differ - compare the <em>notes</em>.</p>
<details><summary>tokens and the synthetic crop (fallback)</summary>
<h2>reference pitches</h2><code>{_sequence(reference)}</code>
<h2>predicted on scan</h2><code>{_sequence(scanned.get('predicted') or [])}</code>
<h2>synthetic crop &mdash; a <em>different</em> system, not a twin</h2>
<img src='/image?track=synthetic&name={_escape(staff)}'>
</details>
<form method='post' action='/judge'>
<input type='hidden' name='name' value='{_escape(staff)}'>
<button name='judgment' value='label-ok'>same music &mdash; label is right</button>
<button name='judgment' value='label-wrong'>different music &mdash; label is wrong</button>
<button name='judgment' value='unclear'>unclear</button>
<input name='note' placeholder='note' value='{_escape(judged.get('note', ''))}'>
</form>
<p>current: {_escape(judged.get('judgment', '—'))}</p>"""


class Handler(BaseHTTPRequestHandler):
    state: ReviewState

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            offset = int((query.get("offset") or ["0"])[0] or 0)
            self._send(render_index(self.state, offset).encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/staff":
            name = safe_staff((query.get("name") or [""])[0])
            body = render_staff(self.state, name) if name else None
            if body is None:
                self._send(b"not found", "text/plain", 404)
                return
            self._send(body.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/image":
            name = safe_staff((query.get("name") or [""])[0])
            track = (query.get("track") or ["scanned"])[0]
            path = self.state.image_path(track, name) if name else None
            if path is None:
                self._send(b"not found", "text/plain", 404)
                return
            self._send(path.read_bytes(), "image/png")
            return
        self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(length).decode())
        name = safe_staff((form.get("name") or [""])[0])
        target = "/"
        if name:
            self.state.save_judgment(
                name, (form.get("judgment") or [""])[0], (form.get("note") or [""])[0]
            )
            following = self.state.next_unjudged(name)
            # Falling back to the index rather than the staff just judged: when nothing
            # is left, "here is the list, you are done" is the useful answer.
            target = f"/staff?name={following}" if following else "/"
        self.send_response(303)
        self.send_header("Location", target)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synthetic-index", type=Path, required=True)
    parser.add_argument("--scanned-index", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--synthetic-predictions", type=Path)
    parser.add_argument("--scanned-predictions", type=Path)
    parser.add_argument(
        "--rendered", type=Path,
        help="Directory of label renderings from render_ossq_labels.py.",
    )
    parser.add_argument("--port", type=int, default=8793)
    args = parser.parse_args()

    Handler.state = ReviewState(
        args.synthetic_index, args.scanned_index, args.judgments,
        args.synthetic_predictions, args.scanned_predictions, args.rendered,
    )
    print(
        f"reviewing {len(Handler.state.staves()):,} paired staves "
        f"at http://localhost:{args.port}/"
    )
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
