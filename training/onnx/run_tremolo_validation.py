"""Persistent local fixed-policy validation over frozen new pages and cached controls."""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pages = json.loads((args.run / "pages.json").read_text())
    selection = json.loads((args.run / "selection-receipt.json").read_text())
    policy = Path("training/onnx/probe_staff_lane_timing.py")
    if sha(policy) != selection["policy_sha256"]:
        raise ValueError("Frozen policy changed")
    jobs = []
    for score in ["sq9146376", "sq7358579", "IMSLP154086", "IMSLP89797"]:
        jobs.append(
            dict(
                name="cached-" + score,
                case=str(args.base / "timing-page-selective-20261007" / score),
                cached=True,
            )
        )
    for score in ["IMSLP154110", "IMSLP578231", "IMSLP621832"]:
        jobs.append(
            dict(
                name="cached-" + score,
                case=str(args.base / "timing-positive-pages-20261007" / score),
                cached=True,
            )
        )
    jobs += [
        dict(name=p["score"], index=i, case=str(args.run / p["score"]), cached=False)
        for i, p in enumerate(pages)
    ]
    write(args.run / "jobs.json", jobs)
    done = {}

    def run(job):
        out = args.run / job["name"]
        with (args.run / (job["name"] + ".log")).open("w") as stream:
            try:
                if not job["cached"]:
                    subprocess.run(
                        [
                            sys.executable,
                            "training/onnx/run_positive_page_timing.py",
                            "--pages",
                            str(args.run / "pages.json"),
                            "--index",
                            str(job["index"]),
                            "--models",
                            str(args.run / "models.json"),
                            "--probe-models",
                            str(args.base / "timing-free-decode-20261007/models.json"),
                            "--out",
                            str(out),
                            "--skip-selective",
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                else:
                    out.mkdir(exist_ok=False)
                subprocess.run(
                    [
                        sys.executable,
                        str(policy),
                        "--legacy-tremolo-rule",
                        "--tremolo-only",
                        "--case",
                        job["case"],
                        "--out",
                        str(out / "tremolo-rule"),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                return dict(status="ok", out=str(out / "tremolo-rule"))
            except Exception as error:
                return dict(status="failed", error=f"{type(error).__name__}: {error}")

    def status():
        write(
            args.run / "watch-status.json",
            dict(
                updated_utc=datetime.now(timezone.utc).isoformat(),
                completed=done,
                pending=[j["name"] for j in jobs if j["name"] not in done],
            ),
        )

    status()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, j): j["name"] for j in jobs}
        for future in as_completed(futures):
            done[futures[future]] = future.result()
            status()
    write(
        args.run / "completion.json",
        dict(jobs=done, policy_sha256=sha(policy), source_sha256=sha(__file__)),
    )


if __name__ == "__main__":
    main()
