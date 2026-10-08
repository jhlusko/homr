"""Turn the owner's completed placement review into a validated import and score copies.

Inputs are frozen: the v4 inventory, the owner's review summary (three text corrections,
92 cases without reported errors, printed lyric lines confirmed correct) and the verified
correction-geometry audit. Nothing is inferred from source scores. Merged-token siblings the
owner did not enumerate (wings, ev) are included as unconfirmed pending rows only.
"""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write

PAGES = ["IMSLP154059", "IMSLP24047", "IMSLP476183", "IMSLP556523"]


def build(inventory, summary, audit):
    rows = {r["id"]: deepcopy(r) for r in inventory}
    confirmed = {r["id"] for r in inventory}
    out_rows = []
    for row in inventory:
        row = deepcopy(row)
        out_rows.append(row)
    by_id = {r["id"]: r for r in out_rows}
    pending, duplicates = set(), set()
    for case in audit["cases"]:
        source = rows[case["id"]]
        for number, proposal in enumerate(case["proposals"]):
            if number == 0:
                target = by_id[case["id"]]
                if not proposal["matches_original_anchor"]:
                    raise ValueError("Reviewed text moved off its reviewed anchor")
                target["text"] = proposal["text"]
                same_slot = [
                    r
                    for r in inventory
                    if r["id"] != case["id"]
                    and r["text"] == proposal["text"]
                    and r["score"] == target["score"]
                    and r["physicalRowOrder"] == target["physicalRowOrder"]
                    and r["generatedNote"] == proposal["note"]
                ]
                if same_slot:
                    # The reviewed word is already proposed on this note: the corrected
                    # row was a duplicate candidate (e.g. a printed separator).
                    duplicates.add(case["id"])
                continue
            extra = deepcopy(source)
            extra["id"] = f'{case["id"]}-{proposal["text"]}'
            extra["text"] = proposal["text"]
            extra["generatedNote"] = proposal["note"]
            out_rows.append(extra)
            by_id[extra["id"]] = extra
            confirmed.add(extra["id"])
            if proposal["pending_user_confirmation"]:
                pending.add(extra["id"])
    decisions = []
    for row in out_rows:
        done = row["id"] not in pending
        decisions.append(
            {
                "id": row["id"],
                "text": row["text"],
                "printedLine": row["physicalRowOrder"],
                "textConfirmed": done,
                "verseConfirmed": done,
                "noteConfirmed": done,
                "excluded": row["id"] in duplicates,
                "score": row["score"],
                "pageSha256": row["pageSha256"],
                "musicXmlSha256": row["musicXmlSha256"],
                "generatedNote": row["generatedNote"],
                "predictions_sha256": row["predictions_sha256"],
            }
        )
    return out_rows, {
        "schemaVersion": "ots.lyric-placement-decisions.v2",
        "automaticAttachment": False,
        "decisions": decisions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--service", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.service))
    from import_reviewed_lyric_copy import import_copy

    base = args.base
    inventory_path = (
        base
        / "guarded-tuplet-20261004/review/lyric-placement-confirmation-20261007-v4/inventory.json"
    )
    summary_path = base / "c1-user-placement-corrections-20261007/review-summary.json"
    audit_path = base / "c1-reviewed-corrections-audit-20261007-v3/report.json"
    inventory = json.loads(inventory_path.read_text())
    summary = json.loads(summary_path.read_text())
    audit = json.loads(audit_path.read_text())
    if sha(inventory_path) != summary["inventorySha256"]:
        raise ValueError("Reviewed inventory changed")
    rows, review = build(inventory, summary, audit)
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "reviewed-inventory.json", rows)
    write(args.out / "reviewed-decisions.json", review)
    decode = base / "c1-runtime-introduction-paired-v2-20261005"
    receipts = []
    for index, score in enumerate(PAGES):
        xml = (decode / f"page-{index}" / "page.musicxml").read_bytes()
        output, plan, receipt = import_copy(rows, review, xml, score)
        folder = args.out / score
        folder.mkdir()
        (folder / "score.musicxml").write_bytes(output)
        write(folder / "validated-plan.json", plan)
        write(folder / "receipt.json", receipt)
        receipts.append(
            {
                "score": score,
                "applied": len(receipt["applied"]),
                "pending": len([p for p in plan["pendingPlacements"] if p["score"] == score]),
            }
        )
    write(
        args.out / "summary.json",
        {
            "basis": "owner statement: three text corrections, other cases and printed lines correct",
            "inventory_sha256": sha(inventory_path),
            "summary_sha256": sha(summary_path),
            "audit_sha256": sha(audit_path),
            "source_sha256": sha(__file__),
            "scores": receipts,
            "automatic_attachment": False,
            "pending_unconfirmed": sorted(
                d["id"] for d in review["decisions"] if not d["noteConfirmed"]
            ),
        },
    )
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
