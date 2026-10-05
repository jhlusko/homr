"""Recover page/voice context from paired crop caches without recognition or reference injection."""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from homr.transformer import vocabulary as v
from training.transformer.audit_cached_joint_decoding import events, projection, sha
from training.transformer.audit_cached_pitch_dedup import ARMS, FIELDS, METRICS, load, load_manifest
from training.transformer.audit_guarded_tuplet_cleanup import policy
from training.transformer.review_cached_joint_decoding import restore


def address(row, alignment):
    name = Path(row["id"]).name
    if row["source"] == "ossq":
        match = re.fullmatch(r"(sq\d+)_(\d+)_(\d+)_(\d+)\.txt", name)
        if not match:
            raise ValueError("Invalid OSSQ crop identity")
        return match[1], int(match[2]), int(match[3]), int(match[4])
    match = re.fullmatch(r"(IMSLP\d+)-sys(\d+)-v(\d+)\.tokens", name)
    if not match:
        raise ValueError("Invalid Lieder crop identity")
    system = int(match[2])
    record = next(r for r in alignment[match[1]]["systems"] if r.get("system") == system)
    if record["status"] != "aligned" or not record.get("boundary_certified"):
        raise ValueError("Unqualified Lieder system boundary")
    return match[1], record["page_image"], system, int(match[3])


def owned_stream(records, field):
    symbols = []
    for row in records:
        restored = restore(row[field])
        for symbol in restored:
            symbol.audit_crop_id = row["id"]
        symbols.extend([*restored, v.EncodedSymbol("newline")])
    return symbols


def owned_rows(symbols, rid):
    chords = v._group_into_chords(symbols)
    selected = [[s for s in chord if getattr(s, "audit_crop_id", None) == rid] for chord in chords]
    result = v._flatten_chords([chord for chord in selected if chord])
    return [{key: getattr(s, key) for key in FIELDS} for s in result]


def meter_declarations(records):
    declarations = []
    for row in records:
        pending = None
        measure = 1
        material = False
        for symbol in row["predicted"]:
            rhythm = symbol["rhythm"]
            if rhythm.startswith("timeSignatureBeats_"):
                pending = int(rhythm.split("_", 1)[1])
            elif rhythm.startswith("timeSignature/"):
                denominator = int(rhythm.split("/", 1)[1])
                declarations.append(
                    {
                        "id": row["id"],
                        "crop_relative_measure": measure,
                        "before_crop_material": not material,
                        "beats": pending,
                        "denominator": denominator,
                        "complete": pending is not None,
                    }
                )
                pending = None
            material |= rhythm.startswith(("note_", "rest_"))
            if "barline" in rhythm or "repeat" in rhythm:
                measure += 1
    return declarations


def music_projection(rows, metric):
    musical = [r for r in events(rows) if r["rhythm"].startswith(("note_", "rest_"))]
    if metric == "symbol":
        return [(r["rhythm"], r["pitch"], r["lift"]) for r in musical]
    if metric in ("pitch", "accidental"):
        musical = [r for r in musical if r["rhythm"].startswith("note_")]
    key = {"rhythm": "rhythm", "pitch": "pitch", "accidental": "lift"}[metric]
    return [r[key] for r in musical]


def boundary(rows):
    material = [r for r in rows if r["rhythm"] not in ("BOS", "EOS", "PAD", "newline")]
    last = material[-1]["rhythm"] if material else None
    return {
        "last_symbol": last,
        "ends_in_predicted_divider": bool(last and ("barline" in last or "repeat" in last)),
        "positions": sorted(
            {r["position"] for r in rows if r["rhythm"].startswith(("note_", "rest_"))}
        ),
    }


def run(args):
    if args.out.exists():
        raise FileExistsError("Refusing to replace recovered context")
    alignment_all = json.loads(args.alignment.read_text())
    if alignment_all.get("model_predictions_used") is not False:
        raise ValueError("Alignment depends on model predictions")
    alignment = alignment_all["scores"]
    witnesses = json.loads((args.previous / "tuplet-witnesses.json").read_text())["witnesses"]
    wanted = {w["id"] for w in witnesses}
    manifest = load_manifest(args.run / "holdout.jsonl")
    # Only Lieder witness scores need alignment; unrelated missing records are not consulted.
    target_scores = {w["score"] for w in witnesses}
    expected = {rid: row for rid, row in manifest.items() if row["score"] in target_scores}
    addresses = {rid: address(row, alignment) for rid, row in expected.items()}
    groups = defaultdict(list)
    by_score_voice = defaultdict(list)
    for rid, addr in addresses.items():
        score, page, system, voice = addr
        groups[score, page, voice].append(rid)
        by_score_voice[score, voice].append(rid)
    for ids in groups.values():
        ids.sort(key=lambda rid: addresses[rid][2])
    for ids in by_score_voice.values():
        ids.sort(key=lambda rid: (addresses[rid][1], addresses[rid][2]))
    target_groups = {addresses[rid][:2] + (addresses[rid][3],) for rid in wanted}
    args.out.mkdir(parents=True)
    caches = {arm: load(args.run / "holdout" / (arm + ".jsonl")) for arm in ARMS}
    for arm in ARMS:
        if set(caches[arm]) != set(manifest):
            raise ValueError("Incomplete full cache coverage")
    for rid, row in expected.items():
        left, right = (caches[arm][rid] for arm in ARMS)
        for key in (
            "source",
            "score",
            "image_sha256",
            "reference_sha256",
            "sidecar_sha256",
            "pixel_sha256",
            "reference",
        ):
            if left[key] != right[key]:
                raise ValueError("Unpaired cache input: " + key)
        for key in ("source", "score", "image_sha256", "reference_sha256", "sidecar_sha256"):
            if left[key] != row[key]:
                raise ValueError("Manifest mismatch: " + key)
        for arm in ARMS:
            for metric in METRICS:
                ref = projection(caches[arm][rid]["reference"], metric)
                got = projection(caches[arm][rid]["predicted"], metric)
                if caches[arm][rid]["metrics"][metric] != {
                    "errors": Levenshtein.distance(ref, got),
                    "reference": len(ref),
                }:
                    raise ValueError("Raw recognition control mismatch")
    cases, group_records = [], []
    for index, group in enumerate(sorted(target_groups, key=str), 1):
        score, page, voice = group
        ids = groups[group]
        systems = [addresses[rid][2] for rid in ids]
        all_page_systems = sorted({addr[2] for addr in addresses.values() if addr[:2] == group[:2]})
        missing = sorted(set(all_page_systems) - set(systems))
        page_arms = {}
        for arm in ARMS:
            records = [caches[arm][rid] for rid in ids]
            declarations = meter_declarations(records)
            variants = {}
            outputs = {}
            for name in ("standard", "no_tuplets", "guard"):
                with policy(name):
                    outputs[name] = v.remove_duplicated_symbols(owned_stream(records, "predicted"))
                variants[name] = {rid: owned_rows(outputs[name], rid) for rid in ids}
            page_arms[arm] = {
                "meter_declarations": declarations,
                "variants": variants,
                "joins": [{"id": r["id"], **boundary(r["predicted"])} for r in records],
            }
            for witness in witnesses:
                if witness["arm"] != arm or witness["id"] not in ids:
                    continue
                rid = witness["id"]
                target_order = ids.index(rid)
                same_page_prior = [
                    d
                    for d in declarations
                    if ids.index(d["id"]) < target_order
                    or (d["id"] == rid and d["before_crop_material"])
                ]
                score_ids = by_score_voice[score, voice]
                prior_ids = score_ids[: score_ids.index(rid)]
                prior_declarations = meter_declarations([caches[arm][i] for i in prior_ids])
                prior_page_declarations = [
                    d for d in prior_declarations if addresses[d["id"]][1] != page
                ]
                metrics = {}
                musical_metrics = {}
                for name, streams in variants.items():
                    metrics[name] = {
                        metric: Levenshtein.distance(
                            projection(caches[arm][rid]["reference"], metric),
                            projection(streams[rid], metric),
                        )
                        for metric in METRICS
                    }
                for name, streams in variants.items():
                    musical_metrics[name] = {
                        metric: Levenshtein.distance(
                            music_projection(caches[arm][rid]["reference"], metric),
                            music_projection(streams[rid], metric),
                        )
                        for metric in METRICS
                    }
                case = {
                    "id": rid,
                    "source": witness["source"],
                    "arm": arm,
                    "kind": witness["kind"],
                    "group": list(group),
                    "system": addresses[rid][2],
                    "same_page_systems": systems,
                    "missing_systems_relative_to_published_page": missing,
                    "latest_same_page_meter": same_page_prior[-1] if same_page_prior else None,
                    "latest_prior_page_meter": (
                        prior_page_declarations[-1] if prior_page_declarations else None
                    ),
                    "isolated_metrics": witness["metrics"],
                    "page_context_metrics": metrics,
                    "note_rest_metrics": musical_metrics,
                    "boundary": boundary(caches[arm][rid]["predicted"]),
                    "same_page_neighbor_ids": ids,
                    "streams": {n: s[rid] for n, s in variants.items()},
                }
                if witness["source"] == "lieder":
                    case["source_measure_window"] = next(
                        r
                        for r in alignment[score]["systems"]
                        if r.get("system") == addresses[rid][2]
                    )
                cases.append(case)
        group_records.append(
            {
                "group": list(group),
                "ids": ids,
                "coverage": {
                    "observed_systems": systems,
                    "published_page_union": all_page_systems,
                    "missing": missing,
                },
                "arms": page_arms,
            }
        )
        print(f"Recovered {index}/{len(target_groups)} page/voice groups", flush=True)
    for record in group_records:
        score, page, _ = record["group"]
        if score.startswith("sq"):
            paths = list((args.workspace / "ossq-omr/scores").glob(f"*/*/{score}.musicxml"))
            if len(paths) != 1:
                raise ValueError("Ambiguous native source identity")
            native = ET.parse(paths[0]).getroot()
            if any("homr" in (s.text or "").lower() for s in native.findall(".//software")):
                raise ValueError("Recognizer output cannot be native source")
            names = [part.findtext("part-name") for part in native.findall("part-list/score-part")]
            record["source_parts"] = names
            record["native_source"] = {
                "path": str(paths[0]),
                "sha256": sha(paths[0]),
                "used_for_inference": False,
                "measure_alignment": "not independently certified",
            }
            image = paths[0].parent / f"images/scanned/original/{score}:{page:04d}.png"
            record["original_page"] = {
                "path": str(image),
                "sha256": sha(image),
                "mapping": (
                    "published crop filename page identity; "
                    "crop bounding boxes and native measure alignment not certified"
                ),
            }
        else:
            image = args.workspace / "homr-artifacts/lieder-v8/_build/pages" / page
            record["original_page"] = {
                "path": str(image),
                "sha256": sha(image),
                "mapping": (
                    "frozen independent boundary-safe alignment; "
                    "source staff/voice ownership remains separate"
                ),
            }
    summary = Counter()
    for case in cases:
        prefix = case["kind"]
        summary[prefix + "/cases"] += 1
        summary[prefix + "/same_page_meter"] += case["latest_same_page_meter"] is not None
        summary[prefix + "/prior_page_meter"] += case["latest_prior_page_meter"] is not None
        for name, metrics in case["note_rest_metrics"].items():
            summary[prefix + "/" + name + "/exact_symbols"] += metrics["symbol"] == 0
    root = Path(__file__).resolve().parents[2]
    report = {
        "summary": dict(summary),
        "cases": len(cases),
        "page_voice_groups": len(group_records),
        "threads": 1,
        "inputs": {
            str(p): sha(p)
            for p in [
                args.alignment,
                args.previous / "tuplet-witnesses.json",
                args.run / "holdout.jsonl",
                *[
                    args.run / "holdout" / (arm + suffix)
                    for arm in ARMS
                    for suffix in (".jsonl", ".identity.json")
                ],
            ]
        },
        "code_sha256": {
            name: sha(root / name)
            for name in [
                "training/transformer/recover_tuplet_voice_context.py",
                "training/transformer/audit_guarded_tuplet_cleanup.py",
                "training/transformer/guarded_tuplet_cleanup.py",
                "homr/transformer/vocabulary.py",
                "homr/staff_parsing.py",
                "homr/transformer/staff2score.py",
                "training/transformer/review_cached_joint_decoding.py",
                "training/transformer/audit_cached_joint_decoding.py",
                "training/transformer/audit_cached_pitch_dedup.py",
            ]
        },
        "scoring": (
            "Original crop scorer retained separately. Note/rest metrics omit key-signature "
            "metadata to avoid repeated-header cleanup confounding tuplets."
        ),
        "scope": (
            "Published-crop page/voice reconstruction, not production segmentation rerun "
            "or independently scored native full-page XML. Prior-page meter is diagnostic "
            "evidence only; production does not carry it. Default policies unchanged."
        ),
    }
    for name, data in (
        ("report.json", report),
        ("cases.json", cases),
        ("page-voices.json", group_records),
    ):
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "previous", "workspace", "alignment", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
