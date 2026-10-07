"""Fresh image-first accepted-stack smoke with explicit paired-tremolo flag states.

CPU pipeline smoke, not a production latency measurement. Independent workers use
one thread each; source labels enter scoring only after all exports are frozen.
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write


def worker(args):
    from homr import main
    from training.onnx import qualify_joint_pages as pipeline

    page = json.loads(args.pages.read_text())[args.index]
    pipeline.POLICY.update(
        HOMR_STEM_ARBITRATION="ossq-v7" if page["source"] == "ossq" else "1",
        HOMR_SLUR_SIDE="ossq-rule" if page["source"] == "ossq" else "0",
        HOMR_PAIRED_TREMOLO_TIMING="1" if args.enabled else "0",
    )
    original = main.generate_xml
    export_times = []

    def capture(arguments, voices, title):
        with (args.out / "final-export-input.pkl").open("wb") as handle:
            pickle.dump((arguments, voices, title), handle)
        start = time.monotonic()
        result = original(arguments, voices, title)
        export_times.append(time.monotonic() - start)
        return result

    main.generate_xml = capture
    pipeline.process(page, json.loads(args.models.read_text()), "accepted", args.out)
    record = json.loads((args.out / "result.json").read_text())
    record.update(
        export_seconds=export_times,
        enabled=args.enabled,
        runtime_sha256={
            str(p): sha(p)
            for p in [Path("homr/tremolo_timing.py"), Path("homr/music_xml_generator.py")]
        },
    )
    write(args.out / "result.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--enabled", action="store_true")
    parser.add_argument("--frozen-run", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.frozen_run is None:
        raise ValueError("Require frozen research exports for exact parity comparison")
    args.out.mkdir(parents=True, exist_ok=False)
    pages = json.loads(args.pages.read_text())
    selected = [p for p in pages if p["score"] in {"IMSLP634835", "sq9631717"}]
    if len(selected) != 2:
        raise ValueError("Require the two fixed positive pages")
    write(args.out / "pages.json", selected)
    hashes = {
        str(p): sha(p)
        for p in [
            Path(__file__),
            Path("homr/tremolo_timing.py"),
            Path("homr/music_xml_generator.py"),
        ]
    }
    for path in hashes:
        (args.out / Path(path).name).write_bytes(Path(path).read_bytes())
    write(
        args.out / "selection-receipt.json",
        {
            "pages_sha256": sha(args.pages),
            "models_sha256": sha(args.models),
            "source_sha256": hashes,
            "selection": (
                "Two already-qualified positive pages; "
                "fresh integration smoke, not untouched accuracy test."
            ),
            "frozen_run": str(args.frozen_run),
        },
    )
    jobs = [
        (i, page["score"], enabled) for i, page in enumerate(selected) for enabled in (False, True)
    ]
    done = {}

    def run(job):
        index, score, enabled = job
        name = score + ("-on" if enabled else "-off")
        command = [
            sys.executable,
            __file__,
            "--worker",
            "--pages",
            str(args.out / "pages.json"),
            "--index",
            str(index),
            "--models",
            str(args.models),
            "--out",
            str(args.out / name),
        ]
        if enabled:
            command.append("--enabled")
        try:
            with (args.out / (name + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            return name, {"status": "ok"}
        except Exception as error:
            return name, {"status": "failed", "error": f"{type(error).__name__}: {error}"}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            name, status = future.result()
            done[name] = status
            write(args.out / "watch-status.json", {"completed": done, "total": len(jobs)})
    if any(sha(p) != digest for p, digest in hashes.items()):
        raise ValueError("Frozen runtime changed during fresh smoke")
    write(args.out / "completion.json", {"jobs": done, "source_sha256": hashes})
    # All XML has now been written before native source references are opened.
    from training.onnx.audit_tremolo_validation import native_measures
    from training.onnx.score_full_page_musicxml import compare, staff_events
    from training.onnx.score_paired_page_diagnostic import read_reference

    cases = []
    for page in selected:
        score = page["score"]
        if any(done[score + suffix]["status"] != "ok" for suffix in ("-off", "-on")):
            cases.append({"score": score, "status": "failed"})
            continue
        roots, timings, matches = {}, {}, {}
        for arm, suffix in [("control", "-off"), ("selective", "-on")]:
            directory = args.out / (score + suffix)
            roots[arm] = ET.parse(directory / "page.musicxml").getroot()
            timings[arm] = json.loads((directory / "result.json").read_text())
            frozen = args.frozen_run / score / "tremolo-rule"
            receipt = json.loads((frozen / "prediction-only-proposals.json").read_text())
            path = frozen / (arm + ".musicxml")
            if sha(path) != receipt["xml_sha256"][arm]:
                raise ValueError("Frozen comparator XML changed")
            matches[arm] = ET.tostring(roots[arm]) == ET.tostring(ET.parse(path).getroot())
        if sha(page["reference"]) != page["reference_sha256"]:
            raise ValueError("Native source reference changed")
        reference = staff_events(
            read_reference(page["reference"]), page["start_measure"], page["end_measure"]
        )
        events = {arm: staff_events(root) for arm, root in roots.items()}
        measures = native_measures(reference, events["control"], events["selective"])
        cases.append(
            {
                "score": score,
                "source": page["source"],
                "status": "ok" if all(matches.values()) else "parity-mismatch",
                "exact_frozen_matches": matches,
                "timings": timings,
                "metrics": {arm: compare(reference, value) for arm, value in events.items()},
                "regressed_native_measures": [m["page_bar"] for m in measures if m["regressed"]],
            }
        )
        write(args.out / (score + "-native-audit.json"), measures)
    write(
        args.out / "report.json",
        {
            "cases": cases,
            "attempted": len(jobs),
            "successful": sum(d["status"] == "ok" for d in done.values()),
            "limitation": (
                "Two known positive pages, fresh CPU image-first smoke. "
                "Parallel one-thread workers; "
                "paired timing differences include contention/cold graph effects, "
                "not a production latency budget. "
                "Title OCR and text detectors disabled, as in accepted comparator."
            ),
        },
    )


if __name__ == "__main__":
    main()
