"""Persistent one-thread reserved-calibration validation with frozen lyric recovery gates."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    pages = root / "inputs/inference-pages.json"
    evaluation = root / "inputs/evaluation-pages.json"
    baseline = root / "gate/area200-coverage0.1"
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    repository = Path(__file__).resolve().parents[2]
    provider = repository.parent / "OurTextScores/services/ots-homr-modal"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repository), str(provider), environment.get("PYTHONPATH", "")]
    )
    stages = [
        ("full-ocr", "pilot_linked_lyric_ocr", ["--pages", pages, "--out", root / "full-ocr"]),
        (
            "positions",
            "pilot_lyric_syllable_positions",
            ["--pages", pages, "--baseline", root / "full-ocr", "--out", root / "positions"],
        ),
        (
            "detector",
            "pilot_lyrics_detector_ocr",
            [
                "--pages",
                pages,
                "--weights",
                args.weights,
                "--ocr-receipt",
                root / "full-ocr/receipt.json",
                "--out",
                root / "regions",
            ],
        ),
        (
            "gate",
            "gate_cached_lyric_lines",
            [
                "--pages",
                pages,
                "--ocr",
                root / "full-ocr",
                "--regions",
                root / "regions",
                "--areas",
                "200",
                "--coverages",
                "0.1",
                "--out",
                root / "gate",
            ],
        ),
        (
            "decode",
            "run_lyric_anchor_pilot",
            ["--pages", pages, "--models", args.models, "--out", root / "decode"],
        ),
        (
            "recovery",
            "recover_selected_lyric_rows",
            [
                "--pages",
                pages,
                "--selected",
                baseline,
                "--all-ocr",
                root / "full-ocr",
                "--decode",
                root / "decode",
                "--out",
                root / "recovered",
            ],
        ),
    ]
    for arm, selected in [("baseline", baseline), ("candidate", root / "recovered")]:
        stages.append(
            (
                f"score-{arm}",
                "score_expanded_lyric_attachments",
                [
                    "--pages",
                    evaluation,
                    "--decode",
                    root / "decode",
                    "--selected",
                    selected,
                    "--positions",
                    root / "positions",
                    "--trace-export",
                    "--out",
                    root / f"score-{arm}",
                ],
            )
        )
    completed = []
    input_hashes = {str(p): sha(p) for p in [pages, evaluation, args.models, args.weights]}
    if args.resume:
        previous = json.loads((root / "completion.json").read_text())
        if previous["success"]:
            raise ValueError("Completed run cannot be resumed")
        identity = json.loads((root / "run-identity.json").read_text())
        if identity["inputs_sha256"] != input_hashes:
            raise ValueError("Resume inputs changed")
        completed = [stage for stage in previous["stages"] if stage["exit_code"] == 0]
        for name in ["completion.json", "progress.json", "watcher-completion.json"]:
            if (root / name).exists():
                shutil.copyfile(root / name, root / f"first-attempt-{name}")
        (root / "completion.json").unlink()
    snapshot = root / ("source-snapshot-resume" if args.resume else "source-snapshot")
    snapshot.mkdir(exist_ok=False)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, snapshot / path.name)
    shutil.copytree(
        provider / "detector_runtime",
        snapshot / "detector_runtime",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    write(
        root / ("resume-identity.json" if args.resume else "run-identity.json"),
        {
            "gates": {
                "min_area": 200,
                "coverage": 0.1,
                "row_overlap_larger_height": 0.5,
                "horizontal_extension_median_heights": 4,
            },
            "partition": (
                "unused scores from calibration; not formal test or verified training-independent"
            ),
            "threshold_retuning": False,
            "threads": 1,
            "source_sha256": {str(p): sha(p) for p in snapshot.glob("*.py")},
            "inputs_sha256": input_hashes,
        },
    )
    for name, module, arguments in stages:
        if name in {stage["stage"] for stage in completed}:
            continue
        write(root / "progress.json", {"active": name, "completed": completed})
        tick = time.monotonic()
        with (root / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", f"training.onnx.{module}", *map(str, arguments)],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        completed.append(
            {"stage": name, "seconds": time.monotonic() - tick, "exit_code": result.returncode}
        )
        if result.returncode:
            write(
                root / "completion.json",
                {"success": False, "failed_stage": name, "stages": completed},
            )
            write(
                root / "progress.json",
                {"active": None, "failed_stage": name, "completed": completed},
            )
            raise SystemExit(result.returncode)
    write(root / "completion.json", {"success": True, "stages": completed})
    write(root / "progress.json", {"active": None, "completed": completed})


if __name__ == "__main__":
    main()
