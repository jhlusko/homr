"""Controlled pitch-dedup ablation on paired raw real-source prediction caches."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from homr.transformer.vocabulary import remove_duplicated_symbols
from training.onnx.ablate_cached_duplicate_cleanup import cleanup_without_pitch_dedup
from training.transformer.audit_cached_joint_decoding import projection, sha
from training.transformer.review_cached_joint_decoding import restore

ARMS = ("accepted_oldcore_v7", "public_joint27")
METRICS = ("symbol", "pitch", "rhythm", "accidental")
FIELDS = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


def cleaned(rows, preserve):
    symbols = restore(rows)
    result = (cleanup_without_pitch_dedup if preserve else remove_duplicated_symbols)(symbols)
    return [{key: getattr(symbol, key) for key in FIELDS} for symbol in result]


def load(path):
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in result or row.get("status") != "ok":
                raise ValueError("Duplicate or failed cache row")
            result[row["id"]] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite audit")
    expected = load_manifest(args.run / "holdout.jsonl")
    caches = {arm: load(args.run / "holdout" / (arm + ".jsonl")) for arm in ARMS}
    for arm in ARMS:
        if set(caches[arm]) != set(expected):
            raise ValueError("Incomplete manifest coverage")
        if not (args.run / "holdout" / (arm + ".FINISHED")).exists():
            raise ValueError("Unfinished arm")
    args.out.mkdir(parents=True)
    totals = defaultdict(Counter)
    counts = defaultdict(Counter)
    scores = defaultdict(set)
    changed = []
    for index, rid in enumerate(expected, 1):
        left, right = (caches[arm][rid] for arm in ARMS)
        for key in (
            "source",
            "score",
            "pixel_sha256",
            "image_sha256",
            "reference_sha256",
            "sidecar_sha256",
            "reference",
        ):
            if left[key] != right[key]:
                raise ValueError("Unpaired input: " + key)
        for key in ("source", "score", "image_sha256", "reference_sha256", "sidecar_sha256"):
            if left[key] != expected[rid][key]:
                raise ValueError("Manifest mismatch: " + key)
        source = left["source"]
        if source not in ("ossq", "lieder"):
            raise ValueError("Unexpected source")
        scores[source].add(left["score"])
        for arm in ARMS:
            row = caches[arm][rid]
            variants = {
                "raw": row["predicted"],
                "standard": cleaned(row["predicted"], False),
                "preserve": cleaned(row["predicted"], True),
            }
            metrics = {}
            key = source + "/" + arm
            counts[key]["crops"] += 1
            for variant, symbols in variants.items():
                metrics[variant] = {}
                for metric in METRICS:
                    reference = projection(row["reference"], metric)
                    errors = Levenshtein.distance(reference, projection(symbols, metric))
                    metrics[variant][metric] = errors
                    totals[key][variant + "/" + metric] += errors
                    if variant == "raw":
                        if row["metrics"][metric] != {
                            "errors": errors,
                            "reference": len(reference),
                        }:
                            raise ValueError("Raw metric replay mismatch")
                        totals[key]["reference/" + metric] += len(reference)
            if variants["standard"] != variants["preserve"]:
                counts[key]["changed_crops"] += 1
                for metric in METRICS:
                    delta = metrics["preserve"][metric] - metrics["standard"][metric]
                    counts[key][
                        metric
                        + ("_improved" if delta < 0 else "_worsened" if delta > 0 else "_unchanged")
                    ] += 1
                changed.append(
                    {
                        "id": rid,
                        "source": source,
                        "score": row["score"],
                        "arm": arm,
                        "metrics": metrics,
                        "reference": row["reference"],
                        "standard": variants["standard"],
                        "preserve": variants["preserve"],
                    }
                )
        if index % 500 == 0:
            print(f"{index}/{len(expected)} paired crops", flush=True)
    root = Path(__file__).resolve().parents[2]
    names = [
        "training/transformer/audit_cached_pitch_dedup.py",
        "homr/transformer/vocabulary.py",
        "training/onnx/ablate_cached_duplicate_cleanup.py",
        "training/transformer/audit_cached_joint_decoding.py",
        "training/transformer/review_cached_joint_decoding.py",
    ]
    report = {
        "paired_crops": len(expected),
        "score_counts": {s: len(v) for s, v in scores.items()},
        "counts": dict(counts),
        "totals": dict(totals),
        "threads": 1,
        "code_sha256": {name: sha(root / name) for name in names},
        "input_sha256": {
            str(p): sha(p)
            for p in [
                args.run / "holdout.jsonl",
                *[
                    args.run / "holdout" / (arm + suffix)
                    for arm in ARMS
                    for suffix in (".jsonl", ".identity.json")
                ],
            ]
        },
        "limitations": (
            "Cached crop diagnostic, not full-page or XML voice qualification. "
            "Existing model-selected labels, not untouched test. "
            "Only pitch dedup differs between standard/preserve; "
            "other cleanup remains enabled. Defaults unchanged."
        ),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "changed-crops.json").write_text(json.dumps(changed) + "\n")
    lines = [
        "# Cached real-source pitch-dedup ablation",
        "",
        report["limitations"],
        "",
        "|Source / arm|Metric|Reference events|Standard errors|Preserve errors|Delta|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, values in sorted(totals.items()):
        for metric in METRICS:
            a, b = values["standard/" + metric], values["preserve/" + metric]
            lines.append(f"|{key}|{metric}|{values['reference/' + metric]}|{a}|{b}|{b-a:+d}|")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report["counts"], indent=2), flush=True)


def load_manifest(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError("Duplicate manifest ID")
        rows[row["id"]] = row
    return rows


if __name__ == "__main__":
    main()
