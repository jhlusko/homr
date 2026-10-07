"""Replay frozen accepted objects under a separately frozen tremolo context candidate."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat-meter-context", action="store_true")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=False)
    jobs = json.loads((args.source_run / "jobs.json").read_text())
    original = json.loads((args.source_run / "completion.json").read_text())
    if any(j["status"] != "ok" for j in original["jobs"].values()):
        raise ValueError("Source batch must be complete and successful")
    policy = Path("training/onnx/probe_staff_lane_timing.py")
    peer = Path("training/onnx/tremolo_peer_span.py")
    hashes = dict(policy_sha256=sha(policy), peer_span_sha256=sha(peer))
    for path in [policy, peer, Path(__file__)]:
        (args.run / path.name).write_bytes(path.read_bytes())
    write(args.run / "jobs.json", jobs)
    write(
        args.run / "selection-receipt.json",
        dict(
            source_run=str(args.source_run),
            source_completion_sha256=sha(args.source_run / "completion.json"),
            original_selection_sha256=sha(args.source_run / "selection-receipt.json"),
            **hashes,
            repeat_meter_context=args.repeat_meter_context,
            limitation="Previously inspected fixed-policy cohort; adaptive diagnostic, not untouched qualification.",
        ),
    )
    done = {}

    def run(job):
        with (args.run / (job["name"] + ".log")).open("w") as log:
            try:
                if (
                    sha(policy) != hashes["policy_sha256"]
                    or sha(peer) != hashes["peer_span_sha256"]
                ):
                    raise ValueError("Frozen candidate changed")
                extra = ["--repeat-meter-context"] if args.repeat_meter_context else []
                subprocess.run(
                    [
                        sys.executable,
                        str(policy),
                        "--legacy-tremolo-rule",
                        "--tremolo-only",
                        "--peer-span-context",
                        "--case",
                        job["case"],
                        "--out",
                        str(args.run / job["name"] / "tremolo-rule"),
                        *extra,
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                return dict(status="ok")
            except Exception as error:
                return dict(status="failed", error=f"{type(error).__name__}: {error}")

    def status():
        write(
            args.run / "watch-status.json",
            dict(completed=done, pending=[j["name"] for j in jobs if j["name"] not in done]),
        )

    status()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, j): j["name"] for j in jobs}
        for future in as_completed(futures):
            done[futures[future]] = future.result()
            status()
    if sha(policy) != hashes["policy_sha256"] or sha(peer) != hashes["peer_span_sha256"]:
        raise ValueError("Frozen candidate changed during replay")
    write(args.run / "completion.json", dict(jobs=done, **hashes, source_sha256=sha(__file__)))


if __name__ == "__main__":
    main()
