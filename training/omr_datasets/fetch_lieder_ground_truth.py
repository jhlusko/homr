"""Fetch OpenScore Lieder per-system measure-count ground truth for IMSLP scores that
have exactly one matching Lieder piece.

`scores.yaml` (from https://github.com/OpenScore/Lieder) has an `imslp` field per
piece (e.g. `imslp: '#396671'`) whose numeric part matches our score IDs directly
(`IMSLP396671`). 56 of the ~341 matches map to *more than one* Lieder piece sharing
one IMSLP source (a multi-song collection PDF) - those are skipped here; splitting
one PDF's pages across several pieces needs a different, harder approach and is
handled by `match_collection_pages.py` instead.

`scores.yaml`'s own `path` field can be stale relative to the live repo - found via
a real 404 (`Hensel,_Fanny_(Mendelssohn)/...`, when the actual current folder is
`Hensel,_Fanny/...`, no `(Mendelssohn)` suffix - a folder rename after `scores.yaml`
was last generated). Fetching the repo's own file tree once and looking up each
piece's `lc{key}.mscx` by filename, rather than reconstructing a URL from
`scores.yaml`'s `path`, is robust to this - the path only feeds ordering
(`match_collection_pages.py`'s `order_key`), never the fetch URL.

The exported `.mxl` (compressed MusicXML) for each piece carries no layout
information at all (no `<print>`/system-break elements) - checked directly against a
real piece before relying on it. The `.mscx` (MuseScore's own native, uncompressed
format) does: each `<Measure>` that starts a new line or page carries a
`<LayoutBreak><subtype>line|page</subtype></LayoutBreak>` child. Splitting a piece's
measures at every LayoutBreak gives an ordered list of per-system measure counts -
spot-checked against one real score (IMSLP396671) before building this for real: 18
systems both ways, exact page-grouping match (3/5/5/5), and 15/18 systems' measure
counts exactly matched what `compare_bar_counts.py`'s detector independently counted
from the actual scan.
"""

# flake8: noqa: T201

import argparse
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

LIEDER_SCORES_YAML_URL = "https://raw.githubusercontent.com/OpenScore/Lieder/main/data/scores.yaml"
LIEDER_TREE_URL = "https://api.github.com/repos/OpenScore/Lieder/git/trees/main?recursive=1"
LIEDER_RAW_BASE = "https://raw.githubusercontent.com/OpenScore/Lieder/main/"
LIEDER_MSCX_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OpenScore/Lieder/main/scores/{path}/lc{key}.mscx"
)
LIEDER_MXL_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/OpenScore/Lieder/main/scores/{path}/lc{key}.mxl"
)
MSCX_FILENAME_RE = re.compile(r"lc(\d+)\.mscx$")
MXL_FILENAME_RE = re.compile(r"lc(\d+)\.mxl$")


def load_lieder_scores(cache_path: Path | None) -> dict:
    if cache_path and cache_path.exists():
        return yaml.safe_load(cache_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(LIEDER_SCORES_YAML_URL) as resp:  # noqa: S310
        data = resp.read()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return yaml.safe_load(data)


def load_lieder_file_tree(cache_path: Path | None) -> dict[str, str]:
    """`lieder_key -> current repo path` for every `.mscx` file, from the repo's own
    git tree - not `scores.yaml`'s own `path` field, which can be stale relative to
    the live repo (see this module's own docstring)."""
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(LIEDER_TREE_URL) as resp:  # noqa: S310
            data = json.loads(resp.read())
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
    by_key = {}
    for entry in data.get("tree", []):
        match = MSCX_FILENAME_RE.search(entry["path"])
        if match:
            by_key[match.group(1)] = entry["path"]
    return by_key


def load_lieder_mxl_tree(cache_path: Path | None) -> dict[str, str]:
    """Same as `load_lieder_file_tree`, but indexing each piece's `.mxl` (real,
    exported MusicXML - has lyrics/dynamics content, unlike `.mscx`, which is
    MuseScore's own native format) instead of its `.mscx`. Kept as its own cache
    file (not merged into `load_lieder_file_tree`'s) since the two are fetched and
    used by different callers - not every caller needs both."""
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(LIEDER_TREE_URL) as resp:  # noqa: S310
            data = json.loads(resp.read())
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
    by_key = {}
    for entry in data.get("tree", []):
        match = MXL_FILENAME_RE.search(entry["path"])
        if match:
            by_key[match.group(1)] = entry["path"]
    return by_key


def fetch_mxl(entry: dict, key: str, file_tree: dict[str, str] | None = None) -> bytes:
    """Same fallback reasoning as `fetch_mscx` - prefer the live repo tree over
    `scores.yaml`'s own possibly-stale `path`. Returns the raw `.mxl` (a zip
    archive), not the unzipped MusicXML inside it - see
    `musicxml_text_ground_truth.py`'s own `unzip_mxl` for that."""
    if file_tree and key in file_tree:
        url = LIEDER_RAW_BASE + urllib.parse.quote(file_tree[key])
    else:
        url = LIEDER_MXL_URL_TEMPLATE.format(path=urllib.parse.quote(entry["path"]), key=key)
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return resp.read()


def match_single_piece_scores(
    lieder: dict, score_ids: list[str]
) -> dict[str, tuple[str, dict]]:
    """`score_id -> (lieder_key, entry)` for every score with exactly one Lieder
    match - a score mapping to more than one piece (a collection) is dropped here,
    not guessed at."""
    by_imslp: dict[str, list[tuple[str, dict]]] = {}
    for key, entry in lieder.items():
        imslp = entry.get("imslp")
        if imslp and imslp.startswith("#"):
            by_imslp.setdefault(imslp[1:], []).append((str(key), entry))

    matched = {}
    for score_id in score_ids:
        numeric = score_id.removeprefix("IMSLP")
        candidates = by_imslp.get(numeric)
        if candidates and len(candidates) == 1:
            matched[score_id] = candidates[0]
    return matched


def fetch_mscx(entry: dict, key: str, file_tree: dict[str, str] | None = None) -> bytes:
    """`file_tree` (from `load_lieder_file_tree`) is tried first - it reflects the
    live repo, unlike `scores.yaml`'s own `path` field, which can be stale (see
    this module's own docstring). Falls back to reconstructing the URL from
    `entry["path"]` only if the key isn't in the tree at all (e.g. `file_tree`
    wasn't loaded), not silently on a 404 from the tree-derived URL - a tree miss
    followed by a path-based 404 both being possible is a real failure to report,
    not something to paper over with a second guess.
    """
    if file_tree and key in file_tree:
        url = LIEDER_RAW_BASE + urllib.parse.quote(file_tree[key])
    else:
        url = LIEDER_MSCX_URL_TEMPLATE.format(path=urllib.parse.quote(entry["path"]), key=key)
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        return resp.read()


def measures_per_system(mscx_bytes: bytes) -> list[list[int]]:
    """Per-page lists of per-system measure counts, in reading order.

    Reads the first `<Staff>` that actually has `<Measure>` children - `.mscx` also
    has near-empty `<Staff>` declarations under `<Part>` earlier in the document
    (instrument metadata, no measures), which a plain `.//Staff` would match first.
    All real staves share the same measure/break structure, so any one of them is
    a valid reference for counting.
    """
    root = ET.fromstring(mscx_bytes)
    staff = next(s for s in root.findall(".//Staff") if s.findall("Measure"))
    measures = staff.findall("Measure")

    pages: list[list[int]] = []
    current_page: list[int] = []
    current_system = 0
    for measure in measures:
        layout_break = measure.find("LayoutBreak")
        subtype = layout_break.findtext("subtype") if layout_break is not None else None
        if subtype in ("line", "page") and current_system > 0:
            current_page.append(current_system)
            current_system = 0
        if subtype == "page":
            pages.append(current_page)
            current_page = []
        current_system += 1
    current_page.append(current_system)
    pages.append(current_page)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--score-ids", type=Path, required=True, help="Text file, one score id per line."
    )
    parser.add_argument(
        "--scores-yaml-cache", type=Path,
        help="Local cache of Lieder's scores.yaml - fetched once, reused after.",
    )
    parser.add_argument(
        "--file-tree-cache", type=Path,
        help="Local cache of the Lieder repo's own file tree - fetched once, reused after.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output dir for per-score JSON.")
    args = parser.parse_args()

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    score_ids = [line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()]
    matched = match_single_piece_scores(lieder, score_ids)
    print(f"{len(matched)}/{len(score_ids)} scores matched to exactly one Lieder piece")

    args.out.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0
    for score_id, (key, entry) in matched.items():
        out_path = args.out / f"{score_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            mscx_bytes = fetch_mscx(entry, key, file_tree)
            pages = measures_per_system(mscx_bytes)
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED ({e})")
            failed += 1
            continue
        out_path.write_text(
            json.dumps(
                {
                    "score_id": score_id,
                    "lieder_key": key,
                    "path": entry["path"],
                    "name": entry.get("name"),
                    "pages": pages,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        ok += 1
    print(f"{ok} fetched, {skipped} already cached, {failed} failed")


if __name__ == "__main__":
    main()
