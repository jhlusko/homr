"""Replay an opt-in spatial-gap proposal across all cached calibration OCR rows."""

import argparse
import json
import sys
from pathlib import Path

from training.onnx.lyric_syllable_positions import propose_spatial_splits
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--anchors", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cases, rows = [], 0
    inputs = {}
    for path in sorted(args.positions.glob("IMSLP*.json")):
        inputs[str(path)] = sha(path)
        page = json.loads(path.read_text())
        for row in page["rows"]:
            if row.get("geometry_skip"):
                continue
            rows += 1
            value = propose_spatial_splits(row["text"], row["characters"])
            if value["proposals"]:
                cases.append(
                    dict(
                        score=page["page"]["score"],
                        box_index=row["box_index"],
                        row=row,
                        proposal=value,
                    )
                )
    report = dict(
        rows_checked=rows,
        changed_rows=len(cases),
        cases=cases,
        input_sha256=inputs,
        min_gap_ratio=2.5,
        code_sha256={
            str(p): sha(p)
            for p in [Path(__file__), Path(__file__).with_name("lyric_syllable_positions.py")]
        },
        source_text_used=False,
        production_qualified=False,
        scope="Calibration-selected spacing hypothesis; unchanged rows are stability controls, not independent verified negatives",
    )
    write(args.out / "report.json", report)
    if args.anchors is not None:
        workspace = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(workspace / "OurTextScores/services/ots-homr-modal"))
        from lyric_attachment import LyricAttachment
        from lyric_attachment_evaluation import score_linked_attachments
        from lyric_attachment_policy import LyricBox, NoteAnchor, attach_nearest_x

        witness = next(c for c in cases if c["score"] == "IMSLP89794" and c["box_index"] == 58)
        data = json.loads((args.positions / "IMSLP89794.json").read_text())
        linkage_path = args.anchors / "source-linkage.json"
        linked = next(
            p for p in json.loads(linkage_path.read_text())["pages"] if p["score"] == "IMSLP89794"
        )
        system = next(s for s in linked["systems"] if s["scan_system"] == 2)
        if not system["exact_written_pitch_duration_sequence"]:
            raise ValueError("Vocal source sequence differs")
        links = system["proposed_links"]
        transform = next(
            t
            for t in json.loads((args.anchors / "page-1/staff-transforms.json").read_text())
            if t["staff_call"] == 2
        )
        heads = sorted(transform["detected_noteheads"])
        if len(heads) != len(links) or any(a[0] >= b[0] for a, b in zip(heads, heads[1:])):
            raise ValueError("Vocal notehead association is ambiguous")
        page_id = data["page"]["id"]
        y_values = [point[1] for point in witness["row"]["box"]]
        anchors = [
            NoteAnchor(
                link["source"]["source_part"],
                link["source"]["source_note_ordinal"],
                link["source"]["onset_quarters"],
                head[0] * transform["original_to_processed_ratio"][0],
                min(y_values),
                max(y_values),
                link["source"]["physical_measure"],
            )
            for link, head in zip(links, heads)
        ]
        # Fixed from the original scan/native lyric sequence, not a text lookup.
        identities = [(7, 3), (7, 4), (8, 1)]
        references = []
        for index, (measure, ordinal) in enumerate(identities):
            native = next(
                link["source"]
                for link in links
                if link["source"]["physical_measure"] == measure
                and link["source"]["source_note_ordinal"] == ordinal
            )
            text = next(lyric["text"] for lyric in native["lyrics"] if lyric["verse"] == "1")
            references.append(
                LyricAttachment(
                    page_id,
                    f"ocr:58:syllable:{index}",
                    text,
                    "1",
                    "P1",
                    measure,
                    ordinal,
                    native["onset_quarters"],
                    1.0,
                )
            )
        boxes = [
            LyricBox(
                f"ocr:58:syllable:{index}",
                row["text"],
                *row["center"],
                "1",
                witness["row"]["confidence"],
            )
            for index, row in enumerate(witness["proposal"]["syllables"])
        ]
        predicted = attach_nearest_x(page_id, boxes, anchors)
        write(
            args.out / "witness-score.json",
            dict(
                metrics=score_linked_attachments(
                    references, predicted, [(page_id, box.detector_box_id) for box in boxes]
                ),
                reference=[r.as_dict() for r in references],
                predicted=[r.as_dict() for r in predicted],
                source_linkage_sha256=sha(linkage_path),
                scope="One selected fused calibration region; oracle system/verse routing and count-exact notehead association; not held-out accuracy",
            ),
        )
    print(
        json.dumps(
            dict(
                rows_checked=rows,
                changed_rows=len(cases),
                changes=[
                    dict(
                        score=c["score"],
                        box_index=c["box_index"],
                        proposals=c["proposal"]["proposals"],
                    )
                    for c in cases
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
