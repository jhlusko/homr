"""Sequential, resumable page diagnostics with persistent progress and receipts."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import score_pair


def run(pages_path, models_path, out):
    pages = json.loads(pages_path.read_text())
    models = json.loads(models_path.read_text())
    root = Path(__file__).resolve().parents[2]
    out.mkdir(parents=True, exist_ok=True)
    source_files = [
        "training/onnx/run_paired_page_diagnostics.py",
        "training/onnx/score_paired_page_diagnostic.py",
        "training/onnx/qualify_joint_pages.py",
        "training/onnx/qualify_joint_export.py",
        "homr/main.py",
        "homr/music_xml_generator.py",
        "homr/transformer/configs.py",
        "homr/transformer/decoder_inference.py",
        "homr/transformer/encoder_inference.py",
        "homr/transformer/staff2score.py",
        "homr/transformer/structured_decode.py",
        "homr/stem_arbitration.py",
        "homr/slur_side.py",
    ]
    identity = {
        "pages_sha256": sha(pages_path),
        "models_sha256": sha(models_path),
        "source_sha256": {name: sha(root / name) for name in source_files},
        "threads": 1,
    }
    receipt_path = out / "run-identity.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != identity:
        raise ValueError("Inputs/code changed; cannot resume this diagnostic")
    snapshot = out / "source-snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in source_files:
        saved = snapshot / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes((root / name).read_bytes())
    write(receipt_path, identity)
    for page in pages:
        for field in ("image", "reference"):
            if sha(page[field]) != page[field + "_sha256"]:
                raise ValueError("Page/reference changed before decode")
    for arm in models.values():
        for record in arm.values():
            if sha(record["path"]) != record["sha256"]:
                raise ValueError("Model graph changed")
    finished, durations, reports = [], [], []
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    total = len(pages) * 2
    tick = time.time()
    for index, page in enumerate(pages):
        directory = out / page["score"]
        for arm in ("accepted", "candidate"):
            for name, digest in identity["source_sha256"].items():
                if sha(root / name) != digest:
                    raise ValueError("Runtime code changed during job: " + name)
            folder = directory / arm
            result_path = folder / "result.json"
            if result_path.exists():
                prior = json.loads((folder / "identity.json").read_text())
                if prior["page"] != page or prior["models"] != models[arm]:
                    raise ValueError("Prior arm identity differs")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable,
                    str(root / "training/onnx/qualify_joint_pages.py"),
                    "--pages",
                    str(pages_path),
                    "--models",
                    str(models_path),
                    "--arm",
                    arm,
                    "--page-index",
                    str(index),
                    "--out",
                    str(folder),
                ]
                started = time.time()
                with (folder / "decode.log").open("w") as log:
                    process = subprocess.Popen(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                    while process.poll() is None:
                        elapsed = time.time() - started
                        mean = sum(durations) / len(durations) if durations else None
                        write(
                            out / "progress.json",
                            {
                                "status": "running",
                                "pid": os.getpid(),
                                "child_pid": process.pid,
                                "score": page["score"],
                                "arm": arm,
                                "completed": len(finished),
                                "planned": total,
                                "active_seconds": elapsed,
                                "updated": time.time(),
                                "eta_seconds": (
                                    max(0, mean * (total - len(finished)) - elapsed)
                                    if mean
                                    else None
                                ),
                            },
                        )
                        time.sleep(5)
                if not result_path.exists():
                    raise RuntimeError(
                        "Worker exited without result receipt; inspect " + str(folder)
                    )
            result = json.loads(result_path.read_text())
            finished.append({"score": page["score"], "arm": arm, **result})
            if result["status"] == "ok":
                durations.append(result["seconds"])
            write(out / "completed-arms.json", finished)
        report = score_pair(page, directory)
        reports.append(report)
    summary = {
        "status": "finished",
        "seconds": time.time() - tick,
        "attempted_arms": total,
        "successful_arms": sum(r["successful_arms"] for r in reports),
        "paired_successful_pages": sum(r["successful_arms"] == 2 for r in reports),
        "pages": reports,
        "run_identity": identity,
        "scope": "Previously evaluated OSSQ diagnostic pages; not final release accuracy/latency",
    }
    write(out / "summary.json", summary)
    write(
        out / "progress.json",
        {
            "status": "finished",
            "completed": total,
            "planned": total,
            "eta_seconds": 0,
            "updated": time.time(),
        },
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pages", "run_identity")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.pages.resolve(), args.models.resolve(), args.out.resolve())
