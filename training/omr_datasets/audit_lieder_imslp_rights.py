"""Build a per-source IMSLP rights-evidence ledger for the releasable Lieder corpus.

The corpus uses images downloaded from individual IMSLP *file* records.  A work can
have several editions with different copyright statuses, so inspecting a composer or
work page alone is insufficient.  This tool resolves every ``IMSLP<n>`` source in
the supplied indexes to its exact rendered IMSLP file block, and pairs that block
with the corresponding source-template fields.

``candidate_pd_scan`` is deliberately a conservative *publication candidate*, not a
legal conclusion: the exact file record says ``Copyright=Public Domain`` and is a
``Normal Scan``.  It excludes new typesettings and every ambiguous/non-PD/missing
record.  A release still needs a human provenance review and a dataset card that
links each retained source record.

The only network endpoint is IMSLP's public MediaWiki API.  Responses are cached so
that the ledger is reproducible and reruns do not repeatedly query IMSLP.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import yaml
from bs4 import BeautifulSoup

API = "https://imslp.org/api.php"
FILE_RE = re.compile(r"\{\{#fte:imslpfile\s*\n(.*?)\n\}\}", re.DOTALL)
FIELD_RE = re.compile(r"^\|\|?\s*([^=\n]+?)\s*=\s*(.*)$", re.MULTILINE)
ID_RE = re.compile(r"IMSLP\d+")


def api(params: dict[str, str]) -> dict:
    request_url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(request_url) as response:  # noqa: S310 - fixed official API
        return json.load(response)


def cached_api(cache: Path, key: str, params: dict[str, str], pause: float) -> dict:
    path = cache / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = api(params)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    time.sleep(pause)
    return data


def score_ids(indexes: list[Path]) -> tuple[set[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for index in indexes:
        for line in index.read_text(encoding="utf-8").splitlines():
            match = ID_RE.search(line)
            if match:
                counts[match.group()] += 1
    return set(counts), counts


def work_title(entry: dict) -> str:
    """Derive IMSLP's conventional work title from Lieder's stable source path."""
    bits = entry["path"].split("/")
    composer = bits[0].replace("_", " ")
    # OpenScore paths are composer / collection-or-_ / individual work.  IMSLP's
    # work page title is normally the individual work, not its collection.
    work = bits[-1].replace("_", " ")
    return f"{work} ({composer})"


def normalized(value: str) -> str:
    """Compare IMSLP/OpenScore names despite apostrophe and Unicode variants."""
    value = value.replace("’", "'").replace("`", "'")
    value = unicodedata.normalize("NFKD", value)
    return "".join(char for char in value.casefold() if not unicodedata.combining(char))


def resolve_work_page(cache: Path, score_id: str, title: str, pause: float) -> tuple[dict, str]:
    """Resolve OpenScore's path title to IMSLP's canonical work-page title.

    The source repo sometimes has a catalogue spelling (``D550d``) or a collection
    ordinal (``4 Fischerweise``) that differs from IMSLP.  ``allpages`` is used only
    as a constrained fallback, matched on its composer suffix and work-title stem.
    """
    query = cached_api(
        cache,
        f"query_{score_id}",
        {
            "action": "query",
            "titles": title,
            "prop": "revisions",
            "rvprop": "content",
            "format": "json",
        },
        pause,
    )
    page = next(iter(query.get("query", {}).get("pages", {}).values()), {})
    if page.get("pageid") and page.get("revisions"):
        return page, title
    work, composer = title.rsplit(" (", 1)
    composer = composer.removesuffix(")")
    stems = [
        work,
        re.sub(r"^\d+\s+", "", work),
        work.split(",", 1)[0],
        re.sub(r"^\d+\s+", "", work).split(",", 1)[0],
    ]
    seen = set()
    for stem in stems:
        if not stem or stem in seen:
            continue
        seen.add(stem)
        listing = cached_api(
            cache,
            f"allpages_{score_id}_{len(seen)}",
            {
                "action": "query",
                "list": "allpages",
                "apprefix": stem.replace("’", "'"),
                "aplimit": "50",
                "format": "json",
            },
            pause,
        )
        for candidate in listing.get("query", {}).get("allpages", []):
            candidate_title = candidate["title"]
            if " (" not in candidate_title:
                continue
            candidate_work, candidate_composer = candidate_title.rsplit(" (", 1)
            if normalized(candidate_composer.removesuffix(")")) != normalized(composer):
                continue
            # Prefix agreement prevents selecting another work by the same composer.
            if not normalized(candidate_work).startswith(normalized(stem)):
                continue
            resolved = cached_api(
                cache,
                f"resolved_{score_id}",
                {
                    "action": "query",
                    "pageids": str(candidate["pageid"]),
                    "prop": "revisions",
                    "rvprop": "content",
                    "format": "json",
                },
                pause,
            )
            resolved_page = next(iter(resolved.get("query", {}).get("pages", {}).values()), {})
            if resolved_page.get("revisions"):
                return resolved_page, candidate_title
    return page, title


def field_map(block: str) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in FIELD_RE.findall(block)}


def extract_rows(score_id: str, title: str, raw: str, html: str) -> tuple[dict | None, str]:
    blocks = [field_map(block) for block in FILE_RE.findall(raw)]
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find(id=score_id)
    if node is None:
        return None, "file_id_not_rendered_on_work_page"
    # The rendered IMSLP file blocks preserve the source-template order.  Use the
    # containing score tab to get the ordinal, then bind it to the raw field block.
    tab = node.find_parent(class_=re.compile(r"jq-ui-tabs"))
    nodes = tab.select("div[id^='IMSLP']") if tab else []
    try:
        ordinal = [item.get("id") for item in nodes].index(score_id)
    except ValueError:
        return None, "file_id_ordinal_not_found"
    if ordinal >= len(blocks):
        return None, "raw_file_block_missing"
    fields = blocks[ordinal]
    return {
        "imslp_id": score_id,
        "work_title": title,
        "file_name": fields.get("File Name 1", ""),
        "file_description": fields.get("File Description 1", ""),
        "image_type": fields.get("Image Type", ""),
        "scanner": fields.get("Scanner", ""),
        "uploader": fields.get("Uploader", ""),
        "date_submitted": fields.get("Date Submitted", ""),
        "publisher_information": fields.get("Publisher Information", ""),
        "copyright": fields.get("Copyright", ""),
        "source_url": (
            f"https://imslp.org/wiki/Special:ImagefromIndex/{score_id.removeprefix('IMSLP')}"
        ),
        "work_url": "https://imslp.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
    }, ""


def classify(row: dict | None, error: str) -> tuple[str, str]:
    if error:
        return "needs_manual_review", error
    assert row is not None  # noqa: S101
    if row["copyright"].strip().casefold() != "public domain":
        return "exclude", "file record is not explicitly Public Domain"
    if row["image_type"].strip().casefold() != "normal scan":
        return "exclude", "not an original Normal Scan"
    if not row["publisher_information"]:
        return "needs_manual_review", "Public Domain tag lacks edition/publisher provenance"
    return "candidate_pd_scan", "IMSLP exact-file record: Public Domain + Normal Scan"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--scores-yaml", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument(
        "--pause", type=float, default=0.35, help="seconds between uncached API calls"
    )
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    ids, crops = score_ids(args.index)
    lieder = yaml.safe_load(args.scores_yaml.read_text(encoding="utf-8"))
    entries: dict[str, dict] = {}
    for entry in lieder.values():
        numeric = str(entry.get("imslp", "")).removeprefix("#")
        if numeric:
            entries[f"IMSLP{numeric}"] = entry

    rows = []
    for number, score_id in enumerate(sorted(ids), start=1):
        entry = entries.get(score_id)
        base = {
            "imslp_id": score_id,
            "crop_count": crops[score_id],
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if not entry:
            base.update(
                status="needs_manual_review", reason="not mapped by OpenScore Lieder", work_title=""
            )
            rows.append(base)
            continue
        title = work_title(entry)
        page, resolved_title = resolve_work_page(args.cache, score_id, title, args.pause)
        page_id = page.get("pageid")
        raw = (page.get("revisions") or [{}])[0].get("*", "")
        if not page_id or not raw:
            base.update(
                work_title=title,
                resolved_work_title=resolved_title,
                status="needs_manual_review",
                reason="IMSLP work page not resolved",
            )
            rows.append(base)
            continue
        parsed = cached_api(
            args.cache,
            f"parse_{score_id}",
            {"action": "parse", "pageid": str(page_id), "prop": "text", "format": "json"},
            args.pause,
        )
        html = parsed.get("parse", {}).get("text", {}).get("*", "")
        row, error = extract_rows(score_id, resolved_title, raw, html)
        status, reason = classify(row, error)
        base.update(row or {"work_title": title})
        base.update(status=status, reason=reason)
        rows.append(base)
        print(f"[{number}/{len(ids)}] {score_id}: {status}")

    columns = sorted({key for row in rows for key in row})
    args.out_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(Counter(row["status"] for row in rows))


if __name__ == "__main__":
    main()
