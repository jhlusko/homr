"""Split one collection PDF's pages among its constituent Lieder pieces.

56 of the ~341 IMSLP-matched scores in this corpus map to *more than one* Lieder
piece sharing the same IMSLP source - a single scanned PDF containing several
songs (e.g. "5 Songs from the Chinese Poets"), each transcribed as its own separate
Lieder piece. `fetch_lieder_ground_truth.py`'s single-piece assumption (the whole
PDF is one piece) doesn't apply to these; this module finds which pages of the PDF
belong to which piece before the same per-page comparison can run per piece.

Two things this reuses rather than re-derives:

- **Piece order within the collection is already in `scores.yaml`'s own `path`
  field** ("2_The_Ghost_Road", "3a_Geistliches_Lied" for a lettered variant) -
  OpenScore Lieder's own within-collection numbering. Checked against every
  collection in the full 1,356-piece corpus: every single piece has this prefix,
  no exceptions - so piece order is read directly, not guessed at.
- **The per-page "signature" is the same one `fetch_lieder_ground_truth.py` builds
  per piece** (`measures_per_system` - per-page groups of per-system measure counts,
  from splitting the piece's own `.mscx` at `LayoutBreak`s). Only the *page*-level
  granularity (how many systems are on each page), not the full per-system bar
  counts, is used for finding page boundaries here - that's already exactly what
  our own detected `imslp_systems_new_repaired` yaml has on disk with no new
  detection needed, and it's discriminating enough to search with (a piece's own
  page-by-page system-count sequence is a fairly distinctive fingerprint already).

The search: for each piece, in order, slide its own per-page system-count sequence
across the whole PDF's own per-page system-count sequence and take the
lowest-total-absolute-difference contiguous window - but only searching *after*
wherever the previous piece's window ended, so a collection's own known ordering
directly constrains the search instead of an independent, orderless match per piece.
If a piece can't be placed with a exact-length window past that point at all (the
PDF ran out of pages), the whole collection is reported unresolved rather than
forcing a partial or overlapping guess - the same "flag it, don't paper over it"
approach `_group_by_geometry` (`homr/staff_parsing.py`) already takes for a staff
group it can't confidently place.
"""

# flake8: noqa: T201

import argparse
import json
import re
from pathlib import Path

import yaml

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mscx,
    load_lieder_file_tree,
    load_lieder_scores,
    measures_per_system,
)

ORDER_PREFIX_RE = re.compile(r"^(\d+)([a-z]?)_")


def collection_entries(lieder: dict) -> dict[str, list[tuple[str, dict]]]:
    """`imslp numeric id -> every (key, entry) sharing it`, for ids with more than
    one match - a single-match id is `fetch_lieder_ground_truth.py`'s own case, not
    this module's."""
    by_imslp: dict[str, list[tuple[str, dict]]] = {}
    for key, entry in lieder.items():
        imslp = entry.get("imslp")
        if imslp and imslp.startswith("#"):
            by_imslp.setdefault(imslp[1:], []).append((str(key), entry))
    return {numeric: entries for numeric, entries in by_imslp.items() if len(entries) > 1}


def order_key(entry: dict) -> tuple[int, str]:
    """Sort key from `path`'s own leading number ("2_The_Ghost_Road" -> `(2, "")`,
    "3a_Geistliches_Lied" -> `(3, "a")`) - every collection piece in the full Lieder
    corpus has this prefix (checked directly, not assumed), so a piece with none is
    a real surprise worth failing loudly on rather than silently mis-ordering.
    """
    last_segment = entry["path"].split("/")[-1]
    match = ORDER_PREFIX_RE.match(last_segment)
    if not match:
        raise ValueError(f"no leading order number in path segment {last_segment!r}")
    return int(match.group(1)), match.group(2)


def page_system_counts(systems_doc: dict) -> list[int]:
    """Per-page system count, in page order, from an already-detected
    `imslp_systems(_repaired)/*.yaml` document - no new detection needed."""
    return [len(systems_doc["pages"][page_number]["systems"]) for page_number in sorted(systems_doc["pages"])]


def piece_page_signature(pages: list[list[int]]) -> list[int]:
    """Per-page system count for one piece, from `measures_per_system`'s own
    per-page-of-per-system-measure-counts output - only the page shape is used here,
    the actual measure counts matter once a page range is found, not before."""
    return [len(page) for page in pages]


def best_window(
    signature: list[int], pdf_counts: list[int], start_from: int
) -> tuple[int, int, int] | None:
    """`(start, end, score)` for the lowest-scoring contiguous window of
    `pdf_counts` (a page range `[start, end)`) matching `signature`, searched only
    from `start_from` onward - lower score is a better match (sum of absolute
    per-page system-count differences, 0 for a perfect match). `None` if
    `signature` doesn't fit anywhere from `start_from` at all (the PDF ran out of
    pages for this piece).
    """
    length = len(signature)
    best: tuple[int, int, int] | None = None
    for start in range(start_from, len(pdf_counts) - length + 1):
        window = pdf_counts[start : start + length]
        score = sum(abs(a - b) for a, b in zip(signature, window, strict=True))
        if best is None or score < best[2]:
            best = (start, start + length, score)
    return best


def match_collection(
    pieces: list[tuple[str, dict, list[list[int]]]], pdf_counts: list[int]
) -> list[dict] | None:
    """`pieces` already in collection order, each `(key, entry, pages)` - `pages`
    is `measures_per_system`'s own return for that piece. Returns one assignment
    dict per piece, in order, or `None` if any piece can't be placed at all past
    where the previous one ended - a real failure to report, not something to
    guess past.
    """
    assignments = []
    cursor = 0
    for key, entry, pages in pieces:
        signature = piece_page_signature(pages)
        result = best_window(signature, pdf_counts, cursor)
        if result is None:
            return None
        start, end, score = result
        assignments.append(
            {
                "lieder_key": key,
                "path": entry["path"],
                "name": entry.get("name"),
                "page_start": start,
                "page_end": end,
                "match_score": score,
                "pages": pages,
            }
        )
        cursor = end
    return assignments


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
    parser.add_argument("--systems", type=Path, required=True, help="imslp_systems(_repaired) dir.")
    parser.add_argument("--out", type=Path, required=True, help="Output dir for per-score JSON.")
    args = parser.parse_args()

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    score_ids = [line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()]
    collections = collection_entries(lieder)

    args.out.mkdir(parents=True, exist_ok=True)
    matched_scores = 0
    resolved = 0
    unresolved = []
    for score_id in score_ids:
        numeric = score_id.removeprefix("IMSLP")
        entries = collections.get(numeric)
        if not entries:
            continue
        matched_scores += 1
        out_path = args.out / f"{score_id}.json"
        if out_path.exists():
            resolved += 1
            continue

        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            print(f"{score_id}: no detected systems file, skipping")
            continue
        systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
        pdf_counts = page_system_counts(systems_doc)

        ordered_entries = sorted(entries, key=lambda pair: order_key(pair[1]))
        pieces = []
        try:
            for key, entry in ordered_entries:
                mscx_bytes = fetch_mscx(entry, key, file_tree)
                pages = measures_per_system(mscx_bytes)
                pieces.append((key, entry, pages))
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED fetching pieces ({e})")
            continue

        assignments = match_collection(pieces, pdf_counts)
        if assignments is None:
            print(f"{score_id}: UNRESOLVED - could not place every piece in order")
            unresolved.append(score_id)
            continue

        out_path.write_text(
            json.dumps({"score_id": score_id, "pieces": assignments}, indent=2),
            encoding="utf-8",
        )
        resolved += 1
        total_score = sum(a["match_score"] for a in assignments)
        print(f"{score_id}: resolved, {len(assignments)} piece(s), total match score {total_score}")

    print()
    print(f"{matched_scores} collection scores found in this corpus")
    print(f"{resolved} resolved, {len(unresolved)} unresolved: {unresolved}")


if __name__ == "__main__":
    main()
