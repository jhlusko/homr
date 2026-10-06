"""Persistently score the expanded packet once its sequential decoder finishes."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training.onnx.qualify_joint_export import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pages", "decode", "selected", "positions", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--trace-export", action="store_true")
    args = parser.parse_args()
    receipt = args.out.with_name(args.out.name + "-watch.json")
    write(
        receipt, {"status": "waiting_for_decode", "decode": str(args.decode), "out": str(args.out)}
    )
    completion = args.decode / "completion.json"
    while not completion.exists():
        time.sleep(5)
    command = [sys.executable, "-m", "training.onnx.score_expanded_lyric_attachments"]
    for name in ("pages", "decode", "selected", "positions", "out"):
        command.extend([f"--{name}", str(getattr(args, name))])
    if args.trace_export:
        command.append("--trace-export")
    write(receipt, {"status": "scoring", "decode_completion": json.loads(completion.read_text())})
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    result = subprocess.run(command, env=environment)
    write(
        receipt,
        {
            "status": "complete" if result.returncode == 0 else "failed",
            "exit_code": result.returncode,
            "report": str(args.out / "report.json"),
            "decode_completion": json.loads(completion.read_text()),
        },
    )


if __name__ == "__main__":
    main()
