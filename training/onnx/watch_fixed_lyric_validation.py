"""Persistent completion watcher; compares frozen baseline/candidate target counters."""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from training.onnx.native_lyric_attachment_scoring import event_key
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    while not (root / "completion.json").exists():
        time.sleep(5)
    receipt = json.loads((root / "completion.json").read_text())
    if not receipt["success"]:
        write(root / "watcher-completion.json", {"success": False, "runner": receipt})
        raise SystemExit(1)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for arm in ["baseline", "candidate"]:
        with (root / f"audit-{arm}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "training.onnx.audit_native_lyric_errors",
                    "--pages",
                    str(root / "inputs/evaluation-pages.json"),
                    "--score",
                    str(root / f"score-{arm}"),
                    "--decode",
                    str(root / "decode"),
                    "--out",
                    str(root / f"audit-{arm}"),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    pages = json.loads((root / "inputs/evaluation-pages.json").read_text())
    results, hashes = [], {}
    for page in pages:
        counters, metrics = {}, {}
        for arm in ["baseline", "candidate"]:
            audit_path = root / f"audit-{arm}" / f"{page['score']}.json"
            rows = json.loads(audit_path.read_text())["targets"]
            counters[arm] = Counter(
                (event_key(r["reference"]), r["reference"]["verse"], r["reference"]["text"])
                for r in rows
                if r["category"] == "exact"
            )
            report_path = root / f"score-{arm}/report.json"
            scored = json.loads(report_path.read_text())
            metrics[arm] = next(
                p["metrics"] for p in scored["pages"] if p["score"] == page["score"]
            )
            hashes.update({str(p): sha(p) for p in [audit_path, report_path]})
        results.append(
            {
                "score": page["score"],
                "reference_targets": page["native_targets"],
                "baseline": metrics["baseline"],
                "candidate": metrics["candidate"],
                "gained_exact_targets": sum(
                    (counters["candidate"] - counters["baseline"]).values()
                ),
                "lost_exact_targets": sum((counters["baseline"] - counters["candidate"]).values()),
            }
        )
    summary = {
        "success": True,
        "pages": results,
        "baseline_exact": sum(
            p["baseline"]["joint_literal_matches_after_oracle_page_verse_permutation"]
            for p in results
        ),
        "candidate_exact": sum(
            p["candidate"]["joint_literal_matches_after_oracle_page_verse_permutation"]
            for p in results
        ),
        "targets": sum(p["reference_targets"] for p in results),
        "gained_exact_targets": sum(p["gained_exact_targets"] for p in results),
        "lost_exact_targets": sum(p["lost_exact_targets"] for p in results),
        "input_sha256": hashes,
        "code_sha256": sha(Path(__file__)),
        "scope": (
            "Reserved unseen calibration scores; fixed rule; oracle verse mapping; not formal test or verified model-training independence"
        ),
    }
    write(root / "comparison.json", summary)
    write(
        root / "watcher-completion.json",
        {"success": True, "comparison_sha256": sha(root / "comparison.json")},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
