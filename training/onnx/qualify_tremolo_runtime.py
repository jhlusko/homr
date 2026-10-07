"""Compare runtime flag-off/on exports to separately frozen research XML.

Reads trusted local accepted-object caches; never supplies source labels to runtime.
"""

import argparse
import copy
import hashlib
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.music_xml_generator import generate_xml
from training.onnx.qualify_joint_export import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for run in args.runs:
        completed = json.loads((run / "completion.json").read_text())
        for job in json.loads((run / "jobs.json").read_text()):
            if completed["jobs"][job["name"]]["status"] != "ok":
                raise ValueError("Incomplete or failed prerequisite: " + job["name"])
            jobs.setdefault(job["case"], (run, job))
    results = []
    for case, (run, job) in jobs.items():
        directory = args.out / job["name"]
        directory.mkdir()
        identity = json.loads((Path(case) / "observed/identity.json").read_text())
        frozen = run / job["name"] / "tremolo-rule"
        receipt = json.loads((frozen / "prediction-only-proposals.json").read_text())
        for arm in ("control", "selective"):
            if sha(frozen / (arm + ".musicxml")) != receipt["xml_sha256"][arm]:
                raise ValueError("Frozen research XML changed")
        cache = Path(case) / "observed/final-export-input.pkl"
        if sha(cache) != receipt["input_sha256"]:
            raise ValueError("Accepted input changed")
        with cache.open("rb") as handle:
            arguments, voices, title = pickle.load(handle)  # noqa: S301
        saved_env = dict(os.environ)
        timings = {}
        roots = {}
        try:
            os.environ.update(identity["policy"])
            for arm, enabled in [("control", False), ("selective", True)]:
                runtime_args = copy.copy(arguments)
                runtime_args.paired_tremolo_timing = enabled
                before = hashlib.sha256(pickle.dumps(voices)).hexdigest() if enabled else None
                start = time.monotonic()
                roots[arm] = generate_xml(runtime_args, voices, title)
                timings[arm] = time.monotonic() - start
                if enabled and hashlib.sha256(pickle.dumps(voices)).hexdigest() != before:
                    raise ValueError("Runtime changed caller input")
                ET.ElementTree(roots[arm]).write(
                    directory / (arm + ".musicxml"), encoding="unicode", xml_declaration=True
                )
            matches = {
                arm: (
                    ET.tostring(roots[arm])
                    == ET.tostring(ET.parse(frozen / (arm + ".musicxml")).getroot())
                )
                for arm in roots
            }
            row = {
                "name": job["name"],
                "source": identity["page"]["source"],
                "score": identity["page"]["score"],
                "status": "ok" if all(matches.values()) else "failed",
                "exact_matches": matches,
                "seconds": timings,
                "input_sha256": sha(cache),
                "research_receipt_sha256": sha(frozen / "prediction-only-proposals.json"),
            }
        except Exception as error:
            row = {
                "name": job["name"],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        finally:
            os.environ.clear()
            os.environ.update(saved_env)
        results.append(row)
        write(args.out / "watch-status.json", {"completed": results, "total": len(jobs)})
    write(
        args.out / "report.json",
        {
            "cases": results,
            "attempted": len(results),
            "successful": sum(r["status"] == "ok" for r in results),
            "source_sha256": sha(__file__),
            "runtime_sha256": {
                str(p): sha(p)
                for p in [Path("homr/music_xml_generator.py"), Path("homr/tremolo_timing.py")]
            },
            "limitation": (
                "Exact parity with frozen research diagnostics; "
                "not a new recognition evaluation or promotion."
            ),
        },
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
