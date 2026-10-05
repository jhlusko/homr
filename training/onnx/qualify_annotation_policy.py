"""Build audit-only versioned note-anchor bundles for changed Lieder crops."""

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path

from homr.transformer.vocabulary import serialized_chords
from training.omr_datasets.label_serialization import write_label_bundle
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.onnx.audit_crop_onsets import export_audit, source_events, traced_parse
from training.onnx.audit_full_page_references import read_mxl
from training.onnx.audit_lieder_rest_stretch import clean_crop
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import (
    NOTE_ANCHOR_POLICY,
    read_tokens,
    token_lines_to_str,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "report.json").exists():
        raise ValueError("Refusing overwrite")
    replay = json.loads(args.replay.read_text())
    rows = []
    counts = Counter()
    start = time.monotonic()
    for score in replay["scores"]:
        crops = [c for c in score["crops"] if not c["exact_legacy_tokens"]]
        if not crops:
            continue
        reference = Path(score["reference"])
        if not sha(reference) == score["reference_sha256"]:
            raise ValueError("Qualification invariant failed")
        root = read_mxl(reference)
        events = source_events(root)
        voices = traced_parse(root)
        for crop in crops:
            symbols = clean_crop(
                voices, crop["start_measure"], crop["end_measure"], crop["voice_index"]
            )
            path = args.out / "crops" / Path(crop["token"]).name
            manifest = write_label_bundle(path, symbols, annotation_policy=NOTE_ANCHOR_POLICY)
            restored = read_tokens(str(path), expected_annotation_policy=NOTE_ANCHOR_POLICY)
            attach_sidecar(path, restored)
            if (
                not token_lines_to_str(restored, annotation_policy=NOTE_ANCHOR_POLICY)
                == path.read_text()
            ):
                raise ValueError("Qualification invariant failed")
            original = [s for c in serialized_chords(symbols) for s in c]
            parsed = [s for c in serialized_chords(restored) for s in c]
            if not len(original) == len(parsed):
                raise ValueError("Qualification invariant failed")
            for a, b in zip(original, parsed, strict=True):
                if not (a.rhythm, a.pitch, a.lift, a.position, a.articulation, a.slur) == (
                    b.rhythm,
                    b.pitch,
                    b.lift,
                    b.position,
                    b.articulation,
                    b.slur,
                ):
                    raise ValueError("Qualification invariant failed")
                if not (a.exact_timing == b.exact_timing and a.notation == b.notation):
                    raise ValueError("Qualification invariant failed")
                if hasattr(a, "_onset_audit_id"):
                    b._onset_audit_id = a._onset_audit_id
            legacy_only = copy.deepcopy(restored)
            for symbol in legacy_only:
                symbol.notation = None
                symbol.exact_timing = None
            exported = export_audit(legacy_only, events, crop["start_measure"])
            if not (not exported.get("error") and (not exported["legacy_endpoint_errors"])):
                raise ValueError("Qualification invariant failed")
            if not (
                not exported["measure_errors"]
                and (not exported["missing_events"])
                and (not exported["duplicate_events"])
            ):
                raise ValueError("Qualification invariant failed")
            if not sha(Path(crop["image"])) == crop["image_sha256"]:
                raise ValueError("Qualification invariant failed")
            counts["crops"] += 1
            counts["verified_symbols"] += len(original)
            counts["timed_symbols"] += sum(s.exact_timing is not None for s in original)
            counts["onset_errors"] += len(exported["onset_errors"])
            counts["literal_phrase_errors"] += len(exported["slur_errors"])
            rows.append(
                {
                    "score": score["score"],
                    "token": str(path),
                    "original_token": crop["token"],
                    "image": crop["image"],
                    "image_sha256": crop["image_sha256"],
                    "splits": crop["splits"],
                    "manifest": manifest,
                    "export": exported,
                }
            )
        print(score["score"], len(crops), flush=True)
    write(
        args.out / "report.json",
        {
            "crops": rows,
            "counts": counts,
            "seconds": time.monotonic() - start,
            "policy": NOTE_ANCHOR_POLICY,
            "replay_sha256": sha(args.replay),
            "tool_sha256": sha(Path(__file__)),
            "limitation": (
                "Audit bundles on original images, no corpus promotion or model "
                "predictions; source sidecars round-trip but are stripped before legacy "
                "export checks."
            ),
        },
    )


if __name__ == "__main__":
    main()
