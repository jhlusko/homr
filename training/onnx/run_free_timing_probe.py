"""Persistent CPU worker supervisor: parity pilot, then full free-decoding score."""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.workers, args.threads) < 1:
        parser.error("Require positive workers/threads")
    control = args.run / "control"
    control.mkdir(exist_ok=True)
    identity = dict(
        workers=args.workers,
        threads=args.threads,
        supervisor_sha256=sha(__file__),
        scorer_sha256=sha(Path(__file__).with_name("qualify_free_timing_probe.py")),
        preparation_sha256=sha(args.run / "preparation.json"),
    )
    path = control / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Supervisor resume identity changed")
    write(path, identity)
    status = dict(phase="starting", success=False)
    try:
        for scope in ("pilot", "full"):
            rows = json.loads((args.run / f"inference-{scope}.json").read_text())
            jobs, logs = [], []
            for part in range(args.workers):
                log = (control / f"{scope}-{part:02d}.log").open("a")
                logs.append(log)
                command = [
                    sys.executable,
                    "-m",
                    "training.onnx.qualify_free_timing_probe",
                    "decode",
                    "--run",
                    str(args.run),
                    "--scope",
                    scope,
                    "--part",
                    str(part),
                    "--parts",
                    str(args.workers),
                    "--threads",
                    str(args.threads),
                ]
                jobs.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            try:
                while True:
                    codes = [job.poll() for job in jobs]
                    if any(code not in (None, 0) for code in codes):
                        raise RuntimeError(f"{scope} worker failed: {codes}; inspect worker logs")
                    records = [
                        json.loads(p.read_text())
                        for p in (args.run / scope).glob("part-*/*.json")
                        if p.name not in ("identity.json", "completion.json")
                    ]
                    remaining = len(rows) - len(records)
                    mean = sum(r["seconds"] for r in records) / len(records) if records else None
                    # Per-worker mean includes pilot's second, unhooked parity decode.
                    active = max(1, sum(code is None for code in codes))
                    eta = remaining * mean / active if mean is not None else None
                    status = dict(
                        phase=scope,
                        success=False,
                        completed_crops=len(records),
                        total_crops=len(rows),
                        worker_exit_codes=codes,
                        mean_completed_crop_seconds=mean,
                        remaining_seconds=eta,
                        updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        session="timing-free-decode",
                        eta_basis="completed crop mean / active workers; initialization and scoring extra",
                    )
                    write(control / "status.json", status)
                    if all(code == 0 for code in codes):
                        break
                    time.sleep(15)
                with (control / f"{scope}-score.log").open("w") as log:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "training.onnx.qualify_free_timing_probe",
                            "score",
                            "--run",
                            str(args.run),
                            "--scope",
                            scope,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            finally:
                for job in jobs:
                    if job.poll() is None:
                        job.terminate()
                for job in jobs:
                    job.wait()
                for log in logs:
                    log.close()
        status.update(
            phase="complete", success=True, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat()
        )
    except Exception as error:
        status.update(
            phase="failed",
            success=False,
            error=f"{type(error).__name__}: {error}",
            updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        raise
    finally:
        write(control / "status.json", status)
        write(control / "completion.json", status)


if __name__ == "__main__":
    main()
