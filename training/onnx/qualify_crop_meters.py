"""Check newly retained crop numerators against independently read source meters."""

import argparse
import json
from pathlib import Path

from homr.transformer.vocabulary import TIME_SIGNATURE_BEATS_PREFIX, serialized_chords
from training.onnx.audit_crop_onsets import source_events, traced_parse
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_lieder_rest_stretch import clean_crop
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import token_lines_to_str


def without_numerators(lines):
    return [line for line in lines if not line.startswith(TIME_SIGNATURE_BEATS_PREFIX)]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--previous", type=Path, required=True)
    cli.add_argument("--current", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError("Refusing overwrite")
    old = json.loads(args.previous.read_text())
    current = json.loads(args.current.read_text())
    previous = {c["token"]: c for s in old["scores"] for c in s["crops"]}
    checked = []
    for score in current["scores"]:
        changed = [
            c
            for c in score["crops"]
            if Path(c["replayed_label"]).read_bytes()
            != Path(previous[c["token"]]["replayed_label"]).read_bytes()
        ]
        if not changed:
            continue
        path = Path(score["reference"])
        if sha(path) != score["reference_sha256"]:
            raise ValueError("Source hash mismatch")
        root = read_mxl(path)
        events = source_events(root)
        voices = traced_parse(root)
        for crop in changed:
            old_lines = Path(previous[crop["token"]]["replayed_label"]).read_text().splitlines()
            new_lines = Path(crop["replayed_label"]).read_text().splitlines()
            if without_numerators(old_lines) != without_numerators(new_lines):
                raise ValueError("Crop changed beyond numerator retention")
            symbols = clean_crop(
                voices, crop["start_measure"], crop["end_measure"], crop["voice_index"]
            )
            if token_lines_to_str(symbols) != "\n".join(new_lines):
                raise ValueError("Current crop does not replay")
            groups = serialized_chords(symbols)
            signatures = []
            for index, group in enumerate(groups):
                if not group[0].rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX):
                    continue
                numerator = int(group[0].rhythm.removeprefix(TIME_SIGNATURE_BEATS_PREFIX))
                following = [s for g in groups[index + 1 :] for s in g]
                note = next(s for s in following if hasattr(s, "_onset_audit_id"))
                event = events[note._onset_audit_id]
                expected = event["time_signature"]
                denominator = next(
                    s.rhythm.split("/")[1]
                    for s in following
                    if s.rhythm.startswith("timeSignature/")
                )
                if expected != {"beats": str(numerator), "beat_type": denominator}:
                    raise ValueError(f"Crop meter disagrees with source: {crop['token']}")
                signatures.append(
                    {
                        "numerator": numerator,
                        "denominator": denominator,
                        "next_event": note._onset_audit_id,
                        "source": event,
                    }
                )
            checked.append(
                {
                    "token": crop["token"],
                    "source": str(path),
                    "source_sha256": sha(path),
                    "signatures": signatures,
                }
            )
    write(
        args.out,
        {
            "changed_crops": checked,
            "count": len(checked),
            "previous_sha256": sha(args.previous),
            "current_sha256": sha(args.current),
            "tool_sha256": sha(Path(__file__)),
            "limitation": (
                "Written meters and exact non-numerator crop parity, not model predictions."
            ),
        },
    )


if __name__ == "__main__":
    main()
