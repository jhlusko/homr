"""Qualify unchanged Borodin training timing and combine frozen Lieder support."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path

from training.architecture.transformer.exact_timing_targets import HEADS, timing_values
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.onnx.bind_ossq_repaired_timing import part_timings, semantic_key, timing_map
from training.onnx.build_ossq_pixel_review import measure_signature
from training.onnx.prepare_page_certification_extension import preview_window
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import read_tokens


def resolve_indexed_window(signatures, segment):
    cuts = segment.findall("part")
    if len(signatures) != len(cuts):
        raise ValueError("Source and segment part counts differ")
    starts, lengths = None, set()
    for actual, cut in zip(signatures, cuts, strict=True):
        wanted = [measure_signature(m) for m in cut.findall("measure")]
        lengths.add(len(wanted))
        matches = {
            i for i in range(len(actual) - len(wanted) + 1) if actual[i : i + len(wanted)] == wanted
        }
        starts = matches if starts is None else starts & matches
    if len(lengths) != 1 or lengths == {0} or not starts or len(starts) != 1:
        raise ValueError("No unique all-part musical source window")
    start = next(iter(starts))
    return start, start + next(iter(lengths))


def combine_support(lieder, group, classes):
    combined = {h: Counter(lieder["groups"]["train/identical"]["heads"][h]) for h in HEADS}
    for head in HEADS:
        combined[head].update(group["heads"][head])
    unsupported = {h: [v for v in classes[h] if not combined[h][v]] for h in HEADS}
    return combined, unsupported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recovery", "target-manifest", "lieder-support", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.target_manifest.read_text())
    lieder = json.loads(args.lieder_support.read_text())
    if manifest["durationPolicy"] != "tick-quantized" or lieder["target_manifest_sha256"] != sha(
        args.target_manifest
    ):
        raise ValueError("Frozen manifest/policy mismatch")
    receipt_path = args.recovery / "borodin-token-replay-classified-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    pixels_path = args.recovery / "borodin-all-pixel-receipt.json"
    pixels = {
        Path(c["public"]).name.split("-")[0]: c
        for c in json.loads(pixels_path.read_text())["cases"]
    }
    first = Path(receipt["cases"][0]["segment"])
    source = first.parent.parent.parent / "sq10675759.musicxml"
    if sha(source) != receipt["source_sha256"]:
        raise ValueError("Source changed")
    root = ET.parse(source).getroot()
    signatures = [
        [measure_signature(m) for m in p.findall("measure")] for p in root.findall("part")
    ]
    train = set(
        line.split(",")[1] for line in (args.recovery / "index_train.txt").read_text().splitlines()
    )
    valid = set(
        line.split(",")[1] for line in (args.recovery / "index_valid.txt").read_text().splitlines()
    )
    if any("sq10675759_" in line for line in valid):
        raise ValueError("Source split overlap")
    args.out.mkdir(parents=True, exist_ok=False)
    bundles = args.out / "crops"
    bundles.mkdir()
    group = dict(
        crops=0,
        events=0,
        missing_timing=0,
        heads={h: Counter() for h in HEADS},
        unseen={h: Counter() for h in HEADS},
        reasons=Counter(),
    )
    cache, records, rejected = {}, [], []
    for case in receipt["cases"]:
        label, segment_path = Path(case["label"]), Path(case["segment"])
        stem = label.stem
        target = bundles / label.name
        relative = "scanned/train/" + label.name
        try:
            if relative not in train:
                raise ValueError("Not in frozen training index")
            pixel = pixels[stem]
            image, old_image = Path(pixel["public"]), Path(pixel["backup"])
            if sha(image) != sha(old_image):
                raise ValueError("Historical/public crop pixels differ")
            if segment_path not in cache:
                try:
                    segment = ET.parse(segment_path).getroot()
                    bounds = resolve_indexed_window(signatures, segment)
                    canonical = preview_window(root, *bounds)
                    maps = [
                        timing_map(part_timings(a), part_timings(b))
                        for a, b in zip(
                            canonical.findall("part"), segment.findall("part"), strict=True
                        )
                    ]
                    cache[segment_path] = (bounds, maps)
                except (ValueError, TypeError, KeyError) as error:
                    cache[segment_path] = str(error)
            if isinstance(cache[segment_path], str):
                raise ValueError(cache[segment_path])
            bounds, maps = cache[segment_path]
            replay = args.recovery / "borodin-token-replay" / label.name
            if sha(label) != sha(replay):
                raise ValueError("Public/replayed token bytes differ")
            payload_path = Path(str(replay) + ".timing.json")
            payload = json.loads(payload_path.read_text())
            if payload["tokenSha256"] != sha(replay):
                raise ValueError("Replay timing hash mismatch")
            for row in payload["symbols"]:
                row["timing"] = maps[case["part"] - 1][semantic_key(row["timing"])]
            target = bundles / label.name
            shutil.copyfile(label, target)
            notation = Path(str(label) + ".notation.json")
            shutil.copyfile(notation, Path(str(target) + ".notation.json"))
            timing_path = Path(str(target) + ".timing.json")
            write(timing_path, payload)
            attach_sidecar(target, read_tokens(target))
        except (ValueError, TypeError, KeyError) as error:
            for path in (
                target,
                Path(str(target) + ".notation.json"),
                Path(str(target) + ".timing.json"),
            ):
                path.unlink(missing_ok=True)
            rejected.append(dict(stem=stem, reason=str(error)))
            continue
        group["crops"] += 1
        for row in payload["symbols"]:
            timing = row["timing"]
            if timing is None:
                group["missing_timing"] += 1
                continue
            group["events"] += 1
            values, reasons = timing_values(
                timing,
                row["identity"][0].startswith("rest"),
                tick_quarters=Fraction(1) / Fraction(timing["source_divisions"]),
            )
            group["reasons"].update(reasons)
            for head, value in values.items():
                destination = "heads" if value in manifest["classes"][head] else "unseen"
                group[destination][head][value] += 1
        records.append(
            dict(
                stem=stem,
                physical_window=bounds,
                segment=str(segment_path),
                segment_sha256=sha(segment_path),
                token_sha256=sha(label),
                replay_timing_sha256=sha(payload_path),
                timing_sha256=sha(timing_path),
                notation_sha256=sha(notation),
                image=str(image),
                image_sha256=sha(image),
                historical_image=str(old_image),
            )
        )
    combined, unsupported = combine_support(lieder, group, manifest["classes"])
    (args.out / "index_train.txt").write_text(
        "".join(r["image"] + "," + str(bundles / (r["stem"] + ".txt")) + "\n" for r in records)
    )
    report = dict(
        ossq_train=group,
        combined_unchanged_train_heads=combined,
        classes_without_combined_training_support=unsupported,
        records=records,
        rejected=rejected,
        source=str(source),
        source_sha256=sha(source),
        lieder_support=str(args.lieder_support),
        lieder_support_sha256=sha(args.lieder_support),
        target_manifest=str(args.target_manifest),
        target_manifest_sha256=sha(args.target_manifest),
        replay_receipt_sha256=sha(receipt_path),
        pixel_receipt_sha256=sha(pixels_path),
        train_index_sha256=sha(args.recovery / "index_train.txt"),
        valid_index_sha256=sha(args.recovery / "index_valid.txt"),
        code_sha256=sha(Path(__file__)),
        limitation="One OSSQ score; unique canonical label windows and identical historical/public pixels. "
        "Historical scan alignment remains distinct from independent boundary certification. "
        "No repaired tranche added to these counts; Lieder changed crops excluded; classes frozen.",
    )
    write(args.out / "report.json", report)
    print(json.dumps(dict(ossq_train=group, rejected=len(rejected), unsupported=unsupported)))


if __name__ == "__main__":
    main()
