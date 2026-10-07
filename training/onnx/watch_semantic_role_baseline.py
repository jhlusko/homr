"""Monitor the fixed lyric detector pass, score once, and publish locally."""

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
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--server-root", type=Path, required=True)
    args = parser.parse_args()
    pages = args.inputs / "inference-pages.json"
    inventory = args.gallery / "inventory.json"
    inputs = [
        pages,
        inventory,
        args.inputs / "full-ocr/receipt.json",
        Path(__file__),
        Path(__file__).with_name("score_semantic_role_baseline.py"),
    ]
    hashes = {str(p): sha(p) for p in inputs}
    write(args.run / "control/monitor-identity.json", dict(inputs_sha256=hashes))
    while True:
        finished = list((args.run / "regions").glob("IMSLP*.json"))
        completed = len(finished)
        status = dict(
            updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            completed_pages=completed,
            total_pages=len(json.loads(pages.read_text())),
            phase="detector",
            watcher_session="c1-semantic-baseline-monitor",
        )
        if completed:
            mean = sum(json.loads(p.read_text())["seconds"] for p in finished) / completed
            status["mean_completed_page_seconds"] = mean
            status["remaining_seconds"] = mean * (status["total_pages"] - completed)
            status["eta_basis"] = (
                "completed-page mean; includes the current partially processed page"
            )
        exit_path = args.run / "control/EXIT"
        if exit_path.exists():
            code = int(exit_path.read_text())
            status.update(phase="scoring" if code == 0 else "failed", detector_exit_code=code)
            write(args.run / "control/status.json", status)
            if code:
                break
            if any(sha(Path(p)) != digest for p, digest in hashes.items()):
                status.update(phase="failed", reason="Monitor inputs/code changed")
                write(args.run / "control/status.json", status)
                break
            output = args.run / "scored"
            command = [
                sys.executable,
                "-m",
                "training.onnx.score_semantic_role_baseline",
                "--inventory",
                str(inventory),
                "--pages",
                str(pages),
                "--gate",
                str(args.run / "gate/area200-coverage0.1"),
                "--gallery",
                str(args.gallery),
                "--out",
                str(output),
            ]
            with (args.run / "control/score.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            status.update(
                phase="complete" if result.returncode == 0 else "failed",
                scoring_exit_code=result.returncode,
            )
            if result.returncode == 0:
                link = args.server_root / "c1-semantic-baseline-20261007"
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(output.resolve(), target_is_directory=True)
                status["summary"] = json.loads((output / "report.json").read_text())["summary"]
                status["review_url"] = "http://localhost:8765/c1-semantic-baseline-20261007/"
            write(args.run / "control/status.json", status)
            break
        write(args.run / "control/status.json", status)
        time.sleep(15)


if __name__ == "__main__":
    main()
