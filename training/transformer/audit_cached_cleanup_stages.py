"""Isolate runtime cleanup stages on paired real-source raw prediction caches."""

import argparse
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from homr.transformer import vocabulary
from homr.transformer.vocabulary import remove_duplicated_symbols
from training.transformer.audit_cached_joint_decoding import projection, sha
from training.transformer.review_cached_joint_decoding import restore

ARMS = ("accepted_oldcore_v7", "public_joint27")
METRICS = ("symbol", "pitch", "rhythm", "accidental")
FIELDS = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


STAGES = {
    "tuplets": "_fix_over_eager_tuplets",
    "lower_staff": "_only_keep_lower_staff_if_there_is_a_clef",
    "pitch": "_remove_duplicated_piches",
    "headers": "_remove_redudant_clefs_keys_and_time_signatures",
}
VARIANTS = {
    "standard": (),
    "no_tuplets": ("tuplets",),
    "no_lower_staff": ("lower_staff",),
    "no_pitch": ("pitch",),
    "no_headers": ("headers",),
    "no_tuplets_pitch": ("tuplets", "pitch"),
    "no_all": tuple(STAGES),
}


@contextmanager
def disabled_stages(stages):
    names = [STAGES[stage] for stage in stages]
    original = {name: getattr(vocabulary, name) for name in names}
    try:
        for name in names:
            setattr(vocabulary, name, lambda value: value)
        yield
    finally:
        for name, function in original.items():
            setattr(vocabulary, name, function)


def cleaned(rows, stages):
    with disabled_stages(stages):
        result = remove_duplicated_symbols(restore(rows))
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
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Refusing to overwrite audit")
    previous = json.loads((args.previous / "report.json").read_text())
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
            variants = {"raw": row["predicted"]}
            variants.update(
                {name: cleaned(row["predicted"], stages) for name, stages in VARIANTS.items()}
            )
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
            for variant in VARIANTS:
                if variant == "standard" or variants[variant] == variants["standard"]:
                    continue
                counts[key][variant + "/changed_crops"] += 1
                for metric in METRICS:
                    delta = metrics[variant][metric] - metrics["standard"][metric]
                    outcome = "improved" if delta < 0 else "worsened" if delta > 0 else "unchanged"
                    counts[key][variant + "/" + metric + "/" + outcome] += 1
            if any(variants[name] != variants["standard"] for name in VARIANTS):
                changed.append(
                    {
                        "id": rid,
                        "source": source,
                        "score": row["score"],
                        "arm": arm,
                        "metrics": metrics,
                        "reference": row["reference"],
                        "variants": variants,
                    }
                )
        if index % 500 == 0:
            print(f"{index}/{len(expected)} paired crops", flush=True)
    root = Path(__file__).resolve().parents[2]
    names = [
        "training/transformer/audit_cached_cleanup_stages.py",
        "homr/transformer/vocabulary.py",
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
            "Single-stage ablations retain all other cleanup; combined variants are explicit. "
            "Standalone crop cleanup differs from concatenated production voice context. "
            "Defaults unchanged."
        ),
    }
    for key, values in totals.items():
        for metric in METRICS:
            for variant, old_variant in (
                ("raw", "raw"),
                ("standard", "standard"),
                ("no_pitch", "preserve"),
            ):
                if (
                    values[variant + "/" + metric]
                    != previous["totals"][key][old_variant + "/" + metric]
                ):
                    raise ValueError("Previous audit control mismatch")
    report["previous_report_sha256"] = sha(args.previous / "report.json")
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "changed-crops.json").write_text(json.dumps(changed) + "\n")
    lines = [
        "# Cached cleanup-stage ablations",
        "",
        report["limitations"],
        "",
        "|Source / arm|Variant|Symbol|Pitch|Rhythm|Accidental|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, values in sorted(totals.items()):
        for variant in ("raw", *VARIANTS):
            scores = "|".join(str(values[variant + "/" + metric]) for metric in METRICS)
            lines.append(f"|{key}|{variant}|{scores}|")
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
