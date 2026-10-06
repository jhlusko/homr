"""Persistent sequential pilot runner with per-page logs and completion receipt."""

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
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_files = [
        "training/onnx/run_lyric_anchor_pilot.py",
        "training/onnx/pilot_lyric_note_anchors.py",
        "training/onnx/lyric_anchor_geometry.py",
        "training/onnx/qualify_joint_pages.py",
        "training/onnx/qualify_joint_export.py",
        "homr/staff_parsing.py",
        "homr/staff_dewarping.py",
        "homr/transformer/vocabulary.py",
    ]
    for name in source_files:
        saved = args.out / "source-snapshot" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, saved)
    write(
        args.out / "run-identity.json",
        dict(
            pages_sha256=sha(args.pages),
            models_sha256=sha(args.models),
            source_sha256={name: sha(root / name) for name in source_files},
            threads=1,
        ),
    )
    results = []
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    for index, page in enumerate(json.loads(args.pages.read_text())):
        start = time.monotonic()
        write(args.out / "progress.json", dict(active=page["score"], completed=results))
        with (args.out / f"page-{index}.log").open("w") as log:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.onnx.pilot_lyric_note_anchors",
                    "--pages",
                    str(args.pages),
                    "--models",
                    str(args.models),
                    "--page-index",
                    str(index),
                    "--out",
                    str(args.out / f"page-{index}"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        results.append(
            dict(
                score=page["score"], exit_code=process.returncode, seconds=time.monotonic() - start
            )
        )
    write(args.out / "progress.json", dict(active=None, completed=results))
    write(
        args.out / "completion.json",
        dict(results=results, success=all(r["exit_code"] == 0 for r in results)),
    )


if __name__ == "__main__":
    main()
