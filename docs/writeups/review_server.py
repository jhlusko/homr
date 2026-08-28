#!/usr/bin/env python3
"""Serve the review page, and accept corrected labels back from it.

`python3 -m http.server` cannot take a correction: it has no write path, so a reviewer
who fixed a label in the editor had nowhere to put it and the fix lived in their
Downloads folder. This is that server plus one endpoint.

**What it deliberately does not do.** It does not touch the corpus. A browser POST is
not an auditable way to mutate training data, and a correction that silently replaced a
label would be indistinguishable from the truncation defect this whole rebuild exists to
undo. Corrections land in their own store with full provenance - who/when is not
available, but what/from-what/hash is - and
`training/omr_datasets/apply_corrections.py` is the separate, explicit step that patches
the corpus, with a dry run by default and a backup of every file it rewrites.

Token conversion is not done here either. `music_xml_string_to_tokens` pulls in torch
through `training_vocabulary`, and the machine running the review UI is not necessarily
the machine that has it. This server keeps the reviewer's MusicXML exactly as they saved
it; the conversion happens where the corpus does.
"""

# flake8: noqa: T201

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

#: A set or item name may only look like the ones the generators emit. Anything else is
#: refused rather than sanitised: these become path segments, and a "cleaned" name that
#: still resolves somewhere unexpected is worse than a rejection.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")

MAX_UPLOAD = 8 * 1024 * 1024


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, corrections: Path, **kwargs):
        self.corrections = corrections
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # quieter: one line per API call, not per asset
        if "/api/" in (self.path or ""):
            super().log_message(fmt, *args)

    # -- helpers ---------------------------------------------------------------
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _store(self, set_name: str) -> Path:
        return self.corrections / set_name

    # -- routes ----------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/corrections":
            query = parse_qs(parsed.query)
            set_name = (query.get("set") or [""])[0]
            if not SAFE_NAME.match(set_name):
                return self._json(400, {"error": "bad set name"})
            store = self._store(set_name)
            ids = sorted(p.stem for p in store.glob("*.musicxml")) if store.is_dir() else []
            return self._json(200, {"set": set_name, "corrected": ids})
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/correction":
            return self._json(404, {"error": "no such endpoint"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_UPLOAD:
            return self._json(413, {"error": f"body must be 1..{MAX_UPLOAD} bytes"})
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return self._json(400, {"error": f"bad JSON: {exc}"})

        set_name = str(payload.get("set", ""))
        item_id = str(payload.get("id", ""))
        musicxml = payload.get("musicxml") or ""
        note = str(payload.get("note", ""))[:2000]
        if not SAFE_NAME.match(set_name) or not SAFE_NAME.match(item_id):
            return self._json(400, {"error": "bad set or id"})
        if not musicxml.strip():
            return self._json(400, {"error": "empty MusicXML"})

        # Parse before storing. A file that is not MusicXML - the browser handing over
        # an .mscz, or a truncated download - would otherwise sit in the store looking
        # like a correction until apply_corrections choked on it much later.
        try:
            root = ET.fromstring(musicxml)
        except ET.ParseError as exc:
            return self._json(400, {"error": f"not well-formed XML: {exc}"})
        tag = root.tag.rsplit("}", 1)[-1]
        if tag not in {"score-partwise", "score-timewise"}:
            return self._json(400, {"error": f"root element is <{tag}>, not a MusicXML score"})
        measures = len(root.findall(".//measure"))
        if measures == 0:
            return self._json(400, {"error": "score contains no measures"})

        store = self._store(set_name)
        store.mkdir(parents=True, exist_ok=True)
        target = store / f"{item_id}.musicxml"
        # Never clobber an earlier correction: keep it beside the new one. A reviewer
        # correcting the same item twice is normal, and the first attempt may have been
        # the right one.
        if target.exists():
            previous = sorted(store.glob(f"{item_id}.v*.musicxml"))
            target.rename(store / f"{item_id}.v{len(previous) + 1}.musicxml")
        target.write_text(musicxml, encoding="utf-8")

        record = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "set": set_name,
            "id": item_id,
            "measures": measures,
            "bytes": len(musicxml.encode("utf-8")),
            "sha256": hashlib.sha256(musicxml.encode("utf-8")).hexdigest(),
            "note": note,
            "file": str(target),
        }
        with (self.corrections / "log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"correction stored: {set_name}/{item_id} ({measures} measures)")
        return self._json(200, {"ok": True, **record})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--corrections", type=Path,
                        help="default: <root>/review-data/corrections")
    args = parser.parse_args()

    corrections = args.corrections or (args.root / "review-data" / "corrections")
    corrections.mkdir(parents=True, exist_ok=True)
    handler = partial(ReviewHandler, directory=str(args.root), corrections=corrections)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"serving {args.root} on http://127.0.0.1:{args.port}/review.html")
    print(f"corrections -> {corrections}")
    print("apply them with: python -m training.omr_datasets.apply_corrections --help")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
