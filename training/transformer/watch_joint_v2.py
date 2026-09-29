"""Persist an independent status record for the PDMX-v2 joint training job."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def running(session):
    return subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--session", required=True)
    a = p.parse_args()
    a.run.mkdir(parents=True, exist_ok=True)
    while True:
        now = datetime.now(timezone.utc).isoformat()
        progress = read(a.run / "full" / "progress.json")
        progress_path = a.run / "full" / "progress.json"
        progress_age = time.time() - progress_path.stat().st_mtime if progress_path.exists() else None
        epoch = read(a.run / "full" / "epoch-status.json")
        alive = running(a.session)
        if (a.run / "full" / "COMPLETE").exists():
            state = "complete"
        elif (a.run / "full" / "EARLY_STOPPED.json").exists():
            state = "early_stopped"
        elif alive:
            state = "stalled" if progress_age is not None and progress_age > 15 * 60 else "running"
        else:
            state = "stopped_without_completion_marker"
        record = {"checked_utc": now, "state": state, "tmux_session": a.session,
                  "progress": progress, "last_completed_epoch": epoch,
                  "progress_age_seconds": progress_age,
                  "early_stop": read(a.run / "full" / "EARLY_STOPPED.json")}
        temp = (a.run / "WATCH_STATUS.json.tmp")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(a.run / "WATCH_STATUS.json")
        print(now, state, "updates", progress.get("updates") if progress else None, flush=True)
        if state not in ("running", "stalled"):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
