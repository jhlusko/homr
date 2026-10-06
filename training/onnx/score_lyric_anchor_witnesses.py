"""Three reviewed single-syllable smoke witnesses, with fixed verse/system routing.

This conditional diagnostic does not evaluate automatic lyric-row or verse selection,
multi-syllable OCR splitting, final-export ownership, or corpus accuracy.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from training.onnx.qualify_joint_export import sha, write

WITNESSES = [
    dict(
        score="IMSLP154060",
        box_index=15,
        text="If",
        verse="2",
        system=1,
        physical_measure=6,
        source_note_ordinal=1,
        onset_quarters="0",
        ocr_sha256="d9e001bc04946c2154d6a0e68de9659aed1f2eb9fba7ed59d58b84a960e9de48",
    ),
    dict(
        score="IMSLP89794",
        box_index=10,
        text="Nun",
        verse="1",
        system=0,
        physical_measure=1,
        source_note_ordinal=1,
        onset_quarters="0",
        ocr_sha256="24bf0773f6d87d9780bf184b15b2e93ad2cb8d74fb25dacfe0c6e98d8c8e8fa0",
    ),
    dict(
        score="IMSLP89794",
        box_index=17,
        text="Wein,",
        verse="1",
        system=0,
        physical_measure=3,
        source_note_ordinal=1,
        onset_quarters="0",
        ocr_sha256="24bf0773f6d87d9780bf184b15b2e93ad2cb8d74fb25dacfe0c6e98d8c8e8fa0",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace / "OurTextScores/services/ots-homr-modal"))
    from lyric_attachment import LyricAttachment
    from lyric_attachment_evaluation import score_linked_attachments
    from lyric_attachment_policy import LyricBox, NoteAnchor, attach_nearest_x

    pages = {p["score"]: p for p in json.loads(args.pages.read_text())}
    linkage_path = args.pilot / "source-linkage.json"
    linkage = {p["score"]: p for p in json.loads(linkage_path.read_text())["pages"]}
    references, predictions, detected, records = [], [], [], []
    for witness in WITNESSES:
        page = pages[witness["score"]]
        ocr_path = args.ocr / (witness["score"] + ".json")
        if sha(ocr_path) != witness["ocr_sha256"]:
            raise ValueError("Frozen OCR witness changed")
        ocr = json.loads(ocr_path.read_text())["all_page_text"][witness["box_index"]]
        if ocr["text"] != witness["text"]:
            raise ValueError("Frozen single-syllable witness changed")
        system = next(
            s for s in linkage[witness["score"]]["systems"] if s["scan_system"] == witness["system"]
        )
        if not system["exact_written_pitch_duration_sequence"]:
            raise ValueError("Cannot assign source ownership by index across recognition edits")
        polygon = np.array(ocr["box"])
        x, y = polygon.mean(axis=0)
        # Routing is fixed from the reviewed scan, not inferred by height or text.
        anchors = []
        for link in system["proposed_links"]:
            source, predicted = link["source"], link["prediction"]
            anchors.append(
                NoteAnchor(
                    source_part=source["source_part"],
                    physical_measure=source["physical_measure"],
                    source_note_ordinal=source["source_note_ordinal"],
                    onset_quarters=source["onset_quarters"],
                    center_x=predicted["page_coordinate"][0],
                    staff_top=float(polygon[:, 1].min()),
                    staff_bottom=float(polygon[:, 1].max()),
                )
            )
        box_id = f'full-page-ocr:{witness["box_index"]}'
        reference = LyricAttachment(
            page_id=page["id"],
            detector_box_id=box_id,
            text=witness["text"],
            verse=witness["verse"],
            source_part="P1",
            physical_measure=witness["physical_measure"],
            source_note_ordinal=witness["source_note_ordinal"],
            onset_quarters=witness["onset_quarters"],
            confidence=1.0,
        )
        native = [
            link["source"]
            for link in system["proposed_links"]
            if link["source"]["physical_measure"] == reference.physical_measure
            and link["source"]["source_note_ordinal"] == reference.source_note_ordinal
        ]
        if (
            len(native) != 1
            or {"verse": reference.verse, "text": reference.text} not in native[0]["lyrics"]
        ):
            raise ValueError("Reviewed witness disagrees with native lyrics")
        result = attach_nearest_x(
            page["id"],
            [
                LyricBox(
                    box_id, ocr["text"], float(x), float(y), witness["verse"], ocr["confidence"]
                )
            ],
            anchors,
        )
        references.append(reference)
        predictions.extend(result)
        detected.append((page["id"], box_id))
        records.append(
            dict(
                witness=witness,
                reference=reference.as_dict(),
                predicted=[r.as_dict() for r in result],
            )
        )
    report = dict(
        metrics=score_linked_attachments(references, predictions, detected),
        witnesses=records,
        source_linkage_sha256=sha(linkage_path),
        reviewer="Codex: original scan overlays checked on 2026-10-05",
        scope="Three selected single-syllable smoke controls, oracle system/verse routing; not release accuracy or detector recall",
        automatic_verse_selection_scored=False,
        multi_syllable_boxes_scored=False,
        selected_final_xml_ownership_scored=False,
    )
    write(args.out, report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
