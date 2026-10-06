"""Separate hash-validated scan inference inputs from native evaluation metadata."""

import argparse
import json
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    inference, evaluation = [], []
    root = Path(__file__).resolve().parents[3]
    for selected in json.loads(args.selection.read_text()):
        page = selected["page"]
        if page["partition"] != "calibration" or sha(page["image"]) != page["image_sha256"]:
            raise ValueError("Calibration image identity differs")
        source = root / selected["source"]["source"]
        if sha(source) != selected["source"]["source_sha256"]:
            raise ValueError("Native source identity differs")
        systems = sorted(selected["systems"], key=lambda s: s["system_index"])
        if any(not s["count_exact"] or not s["boundary_certified"] for s in systems):
            raise ValueError("Evaluation needs the existing certified measure windows")
        if [s["system_index"] for s in systems] != list(range(len(systems))):
            raise ValueError("Page-local system indices are not contiguous")
        minimal = {key: page[key] for key in ("id", "score", "partition", "image", "image_sha256")}
        inference.append(minimal)
        evaluation.append(
            {
                **minimal,
                "source": selected["source"],
                "start_measure": selected["start_measure"],
                "end_measure": selected["end_measure"],
                "systems": systems,
                "native_targets": selected["native_targets"],
                "native_verses": selected["native_verses"],
                "note_pixel_ownership_certified": False,
            }
        )
    write(args.out / "inference-pages.json", inference)
    write(args.out / "evaluation-pages.json", evaluation)
    write(
        args.out / "receipt.json",
        {
            "selection_sha256": sha(args.selection),
            "code_sha256": sha(Path(__file__)),
            "pages": len(inference),
            "native_targets": sum(p["native_targets"] for p in evaluation),
            "inference_receives_native_reference": False,
        },
    )
    print(f"Prepared {len(inference)} scan-only inputs")


if __name__ == "__main__":
    main()
