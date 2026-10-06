"""One-thread local runtime regression check for piano-introduction context.

Preserves original outputs. XML equality is a regression check, not accuracy.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--ossq-prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    sources = {}
    for name in (
        "homr/staff_parsing.py",
        "homr/main.py",
        "homr/music_xml_generator.py",
        "training/onnx/qualify_intro_context.py",
    ):
        destination = args.out / "source-snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
        sources[name] = sha(destination)
    write(
        args.out / "identity.json",
        dict(
            sources=sources,
            models_sha256=sha(args.models),
            threads=1,
            scope="Local runtime regression; calibration",
        ),
    )
    pages = json.loads(args.calibration.read_text())
    baseline = [args.prior / f"page-{i}" for i in range(len(pages))]
    pages.append(json.loads((args.ossq_prior / "identity.json").read_text())["page"])
    baseline.append(args.ossq_prior)
    manifest = args.out / "pages.json"
    write(manifest, pages)
    results = []
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, page in enumerate(pages):
        folder = args.out / f"page-{index}"
        tick = time.monotonic()
        command = [
            sys.executable,
            "-m",
            (
                "training.onnx.pilot_lyric_note_anchors"
                if index < len(pages) - 1
                else "training.onnx.qualify_joint_pages"
            ),
            "--pages",
            str(manifest),
            "--models",
            str(args.models),
            "--arm",
            "candidate",
            "--page-index",
            str(index),
            "--out",
            str(folder),
        ]
        with (args.out / f"page-{index}.log").open("w") as log:
            completed = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        record = dict(
            score=page["score"],
            exit_code=completed.returncode,
            seconds=time.monotonic() - tick,
            baseline=str(baseline[index]),
            candidate=str(folder),
        )
        if completed.returncode == 0:
            before = ET.parse(baseline[index] / "page.musicxml").getroot()
            after = ET.parse(folder / "page.musicxml").getroot()
            record["xml_identical"] = ET.tostring(before) == ET.tostring(after)
            record["parts"] = [
                dict(
                    measures=len(p.findall("measure")),
                    pitched_notes=len(p.findall("./measure/note/pitch")),
                )
                for p in after.findall("part")
            ]
        results.append(record)
        write(
            args.out / "progress.json",
            dict(completed=len(results), total=len(pages), results=results),
        )
    write(
        args.out / "completion.json",
        dict(
            results=results,
            all_succeeded=all(r["exit_code"] == 0 for r in results),
            controls_unchanged=all(r.get("xml_identical", False) for r in results[1:]),
        ),
    )
    if any(r["exit_code"] != 0 for r in results):
        raise RuntimeError("One or more runtime comparisons failed; inspect page logs")


if __name__ == "__main__":
    main()
