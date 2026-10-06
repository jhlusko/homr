"""Conditional grouped-lyric comparison with reviewed system/verse routing."""

import argparse
import json
import sys
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write

# Original scan/native references, independent of the positional OCR replay.
CASES = [
    (16, 1, "2", [(6, 2), (6, 3)]),
    (18, 1, "2", [(7, 1), (7, 2), (7, 4)]),
    (20, 1, "1", [(6, 1), (6, 2), (6, 3)]),
    (21, 1, "1", [(6, 4), (7, 1)]),
    (22, 1, "1", [(7, 2), (7, 4)]),
    (30, 2, "2", [(10, 1), (10, 2), (10, 3), (10, 4), (11, 1)]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(workspace / "OurTextScores/services/ots-homr-modal"))
    from lyric_attachment import LyricAttachment
    from lyric_attachment_evaluation import score_linked_attachments
    from lyric_attachment_policy import LyricBox, NoteAnchor, attach_nearest_x

    position_path = args.positions / "IMSLP154060.json"
    positions = json.loads(position_path.read_text())
    linkage_path = args.anchors / "source-linkage.json"
    linkage = next(
        p for p in json.loads(linkage_path.read_text())["pages"] if p["score"] == "IMSLP154060"
    )
    transform_path = args.anchors / "page-0/staff-transforms.json"
    transforms = json.loads(transform_path.read_text())
    references, detected = [], []
    predictions = {"attention": [], "detected_noteheads": []}
    cases = []
    for box_index, system_index, verse, reference_notes in CASES:
        row = positions["rows"][box_index]
        if row.get("geometry_skip") or not row["cached_text_and_box_unchanged"]:
            raise ValueError("Frozen positional OCR row is unusable")
        system = next(s for s in linkage["systems"] if s["scan_system"] == system_index)
        if not system["exact_written_pitch_duration_sequence"]:
            raise ValueError("Source note sequences differ")
        links = system["proposed_links"]
        transform = next(t for t in transforms if t["staff_call"] == system["staff_call"])
        noteheads = sorted(transform["detected_noteheads"])
        if len(noteheads) != len(links) or any(
            a[0] >= b[0] for a, b in zip(noteheads, noteheads[1:])
        ):
            raise ValueError("Cannot link segmentation notes by count/monotonic order")
        source = {
            (link["source"]["physical_measure"], link["source"]["source_note_ordinal"]): link[
                "source"
            ]
            for link in links
        }
        page_id = positions["page"]["id"]
        bounds = [point[1] for point in row["box"]]
        anchors = {arm: [] for arm in predictions}
        for link, notehead in zip(links, noteheads):
            native = link["source"]
            for arm in predictions:
                x = (
                    link["prediction"]["page_coordinate"][0]
                    if arm == "attention"
                    else notehead[0] * transform["original_to_processed_ratio"][0]
                )
                anchors[arm].append(
                    NoteAnchor(
                        source_part=native["source_part"],
                        physical_measure=native["physical_measure"],
                        source_note_ordinal=native["source_note_ordinal"],
                        onset_quarters=native["onset_quarters"],
                        center_x=x,
                        staff_top=min(bounds),
                        staff_bottom=max(bounds),
                    )
                )
        expected = []
        for index, identity in enumerate(reference_notes):
            native = source[identity]
            lyrics = [lyric["text"] for lyric in native["lyrics"] if lyric["verse"] == verse]
            if len(lyrics) != 1:
                raise ValueError("Ambiguous native lyric reference")
            expected.append(
                LyricAttachment(
                    page_id=page_id,
                    detector_box_id=f"ocr:{box_index}:syllable:{index}",
                    text=lyrics[0],
                    verse=verse,
                    source_part=native["source_part"],
                    physical_measure=native["physical_measure"],
                    source_note_ordinal=native["source_note_ordinal"],
                    onset_quarters=native["onset_quarters"],
                    confidence=1.0,
                )
            )
        references.extend(expected)
        boxes = [
            LyricBox(
                f"ocr:{box_index}:syllable:{i}",
                syllable["text"],
                *syllable["center"],
                verse,
                row["confidence"],
            )
            for i, syllable in enumerate(row["syllables"])
        ]
        detected.extend((page_id, box.detector_box_id) for box in boxes)
        record = dict(
            box_index=box_index,
            raw_text=row["text"],
            reference=[r.as_dict() for r in expected],
            predictions={},
        )
        for arm in predictions:
            attached = attach_nearest_x(page_id, boxes, anchors[arm])
            predictions[arm].extend(attached)
            record["predictions"][arm] = [r.as_dict() for r in attached]
        cases.append(record)
    report = dict(
        scorer_sha256=sha(Path(__file__)),
        metrics={
            arm: score_linked_attachments(references, rows, detected)
            for arm, rows in predictions.items()
        },
        cases=cases,
        input_sha256={str(p): sha(p) for p in [position_path, linkage_path, transform_path]},
        source_text_passed_to_ocr=False,
        system_and_verse_routing="Fixed reviewed calibration references",
        segmentation_linkage="Count-exact strictly monotonic vocal noteheads and written-pitch/duration sequence; calibration proposal only",
        scope="Six selected Chaminade lyric regions; conditional first-pass attachment diagnostic, not full C1 qualification",
    )
    write(args.out, report)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
