"""Check user-selected lyric character geometry against frozen runtime note anchors."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.lyric_syllable_positions import (
    reviewed_substring_position,
    split_positioned_text,
)
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_expanded_lyric_attachments import note_events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    inventory_path = (
        base
        / "guarded-tuplet-20261004/review/lyric-placement-confirmation-20261007-v3/inventory.json"
    )
    inventory = {r["id"]: r for r in json.loads(inventory_path.read_text())}
    summary_path = base / "c1-user-placement-corrections-20261007/review-summary.json"
    summary = json.loads(summary_path.read_text())
    if sha(inventory_path) != summary["inventorySha256"]:
        raise ValueError("Reviewed inventory changed")
    decode = base / "c1-runtime-introduction-paired-v2-20261005"
    previous = json.loads((decode / "score-four-runtime/report.json").read_text())
    for path, digest in previous["input_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Frozen runtime input changed")
    pages = ["IMSLP154059", "IMSLP24047", "IMSLP476183", "IMSLP556523"]
    cases, controls = [], []
    for score in pages:
        positions = json.loads(
            (base / f"c1-fixed-rule-validation-20261005/positions/{score}.json").read_text()
        )
        for row in positions["rows"]:
            if row.get("geometry_skip"):
                continue
            new = split_positioned_text(row["text"], row["characters"])
            if new != row.get("syllables", []):
                controls.append(
                    {
                        "score": score,
                        "box_index": row["box_index"],
                        "before": row.get("syllables", []),
                        "after": new,
                    }
                )
    for correction in summary["corrections"]:
        item = inventory[correction["id"]]
        score = item["score"]
        folder = decode / f"page-{pages.index(score)}"
        if sha(folder / "page.musicxml") != item["musicXmlSha256"]:
            raise ValueError("Generated XML changed")
        prediction_path = decode / f"score-four-runtime/{score}-predictions.json"
        if sha(prediction_path) != item["predictions_sha256"]:
            raise ValueError("Cached predictions changed")
        pred = json.loads(prediction_path.read_text())
        original = pred["predictions"][int(item["id"].split(":")[-1])]
        traces = [
            t
            for t in pred["attachment_trace"]
            if t.get("outcome") == "attached"
            and t["selected_row_index"] == original["selected_row_index"]
            and t["syllable_index"] == original["syllable_index"]
        ]
        if len(traces) != 1:
            raise ValueError("Nonunique trace")
        trace = traces[0]
        positions = json.loads(
            (base / f"c1-fixed-rule-validation-20261005/positions/{score}.json").read_text()
        )
        rows = [r for r in positions["rows"] if r["box"] == trace["box"]]
        if len(rows) != 1:
            raise ValueError("Nonunique positioned row")
        row = rows[0]
        words = [correction["reviewedText"]]
        if correction.get("separateText"):
            words.append(correction["separateText"])
        system = next(s for s in pred["systems"] if s["staff_call"] == trace["staff_call"])
        final = [
            r
            for r in json.loads((folder / "final-note-anchors.json").read_text())
            if r["voice_index"] == pred["vocal_context"]["voice_index"]
        ]
        part_index = pred["vocal_context"]["voice_index"]
        exported = note_events(
            ET.parse(folder / "page.musicxml").getroot().findall("part")[part_index]
        )
        proposals = []
        for word in words:
            geometry = reviewed_substring_position(row["text"], row["characters"], word)
            distances = sorted(
                (abs(geometry["center"][0] - x), j) for j, x in enumerate(system["head_xs"])
            )
            if len(distances) > 1 and abs(distances[0][0] - distances[1][0]) < 1e-9:
                raise ValueError("Ambiguous nearest head")
            final_index = system["head_to_final_note"].get(str(distances[0][1]))
            if final_index is None:
                raise ValueError("No accepted runtime note association")
            origin = [final[final_index]["voice_index"], final[final_index]["token_index"]]
            records = [
                r
                for r in pred["export_trace"]["records"]
                if r["part_index"] == part_index and r["origin"] == origin
            ]
            if len(records) != 1:
                raise ValueError("Nonunique XML origin")
            note = exported[records[0]["exported_index"]]
            proposals.append(
                {
                    "text": word,
                    "geometry": geometry,
                    "note": note,
                    "distance_pixels": distances[0][0],
                    "matches_original_anchor": note == item["generatedNote"],
                }
            )
        cases.append(
            {
                "id": item["id"],
                "originalText": item["text"],
                "proposals": proposals,
                "reviewed_text_used": True,
                "verse_identity_inferred": False,
            }
        )
    args.out.mkdir(parents=True, exist_ok=False)
    write(
        args.out / "report.json",
        {
            "cases": cases,
            "source_free_separator_changes": controls,
            "musicxml_modified": False,
            "automatic_attachment": False,
            "scope": "User-selected correction geometry; not an independent recognition result",
            "summary_sha256": sha(summary_path),
            "inventory_sha256": sha(inventory_path),
            "source_sha256": sha(__file__),
            "geometry_source_sha256": sha(Path(__file__).with_name("lyric_syllable_positions.py")),
        },
    )
    (args.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
