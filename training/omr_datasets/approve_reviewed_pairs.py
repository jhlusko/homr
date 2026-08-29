"""Whitelist exactly-clean pairs from a downloaded review export.

Review exports are intentionally separate from a corpus manifest: they are a human
decision log, not an instruction to accept an entire score or unreviewed neighbours.
This utility keeps only individual pair IDs explicitly marked ``complete`` with no note.
It is deliberately conservative: a reviewer who writes "otherwise correct, but the time
signature is wrong" has found a label defect, not approved a training pair.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path


def reviewed_id(row: dict) -> str:
    """Use the stable exported ID, with a compatibility fallback for older exports."""
    if row.get("id"):
        return str(row["id"])
    return f"{row['score_id']}-sys{row['system']}-v{row['voice']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    review = json.loads(args.review.read_text(encoding="utf-8"))
    rows = review.get("reviewed", [])
    clean_ids = {
        reviewed_id(row)
        for row in rows
        if row.get("verdict") == "complete" and not str(row.get("notes") or "").strip()
    }
    kept: list[str] = []
    seen_ids: set[str] = set()
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stem = Path(line.split(",", 1)[0]).stem
        if stem in clean_ids:
            kept.append(line)
            seen_ids.add(stem)

    missing = sorted(clean_ids - seen_ids)
    if missing:
        raise SystemExit(f"review IDs missing from manifest: {', '.join(missing[:10])}")
    args.out.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    nonclean = [
        {
            "id": reviewed_id(row),
            "verdict": row.get("verdict"),
            "notes": row.get("notes"),
        }
        for row in rows
        if reviewed_id(row) not in clean_ids
    ]
    args.report.write_text(
        json.dumps(
            {
                "review": str(args.review),
                "source_manifest": str(args.manifest),
                "policy": "only explicit complete verdicts with an empty note",
                "reviewed_rows": len(rows),
                "approved_pairs": len(kept),
                "not_approved": nonclean,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(kept)} explicitly clean reviewed pair(s) -> {args.out}")
    print(f"{len(nonclean)} reviewed item(s) withheld -> {args.report}")


if __name__ == "__main__":
    main()
