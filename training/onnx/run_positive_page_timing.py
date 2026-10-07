"""Build an accepted control then observe/replay timing, in isolated processes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--probe-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument(
        "--skip-selective",
        action="store_true",
        help="Capture full accepted objects without the separate timing-head policy",
    )
    args = parser.parse_args()
    page = json.loads(args.pages.read_text())[args.index]
    if args.control_only:
        from training.onnx import qualify_joint_pages as worker

        worker.POLICY.update(
            HOMR_STEM_ARBITRATION="ossq-v7" if page["source"] == "ossq" else "1",
            HOMR_SLUR_SIDE="ossq-rule" if page["source"] == "ossq" else "0",
        )
        worker.process(page, json.loads(args.models.read_text()), "accepted", args.out / "accepted")
        return
    if args.out.exists():
        raise ValueError("Output must be new")
    args.out.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--pages",
            str(args.pages),
            "--index",
            str(args.index),
            "--models",
            str(args.models),
            "--probe-models",
            str(args.probe_models),
            "--out",
            str(args.out),
            "--control-only",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "training/onnx/observe_page_timing_probe.py",
            "--baseline",
            str(args.out / "accepted"),
            "--probe-models",
            str(args.probe_models),
            "--out",
            str(args.out / "observed"),
        ],
        check=True,
    )
    if args.skip_selective:
        return
    command = [
        sys.executable,
        "training/onnx/replay_selective_page_timing.py",
        "--observation",
        str(args.out / "observed"),
        "--out",
        str(args.out / "selective"),
    ]
    if not page["boundary_certified"]:
        command.append("--skip-reference-scoring")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
