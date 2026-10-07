"""Research-only image-first timing probe; references never enter greedy decoding."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from homr.transformer.vocabulary import kern_to_symbol_duration
from training.architecture.transformer.exact_timing_targets import HEADS, TYPE_QUARTERS
from training.onnx.qualify_joint_export import sha, write

DIVIDERS = {
    "barline",
    "doublebarline",
    "bolddoublebarline",
    "repeatStart",
    "repeatEnd",
    "repeatBoth",
    "repeatEndStart",
}
FIELDS = ("rhythm", "pitch", "lift", "articulation", "slur", "position")


def serialize(symbol):
    return {field: getattr(symbol, field) for field in FIELDS}


def measures(symbols):
    groups = [[]]
    for index, symbol in enumerate(symbols):
        rhythm = symbol["rhythm"]
        if rhythm in DIVIDERS:
            groups.append([])
        elif rhythm.startswith("note"):
            groups[-1].append((index, ("note", symbol["pitch"], symbol["lift"], "G" in rhythm)))
        elif rhythm.startswith("rest"):
            groups[-1].append((index, ("rest", rhythm.endswith("m"))))
        elif rhythm == "chord":
            groups[-1].append((index, ("chord",)))
    return groups


def match_events(reference, predicted):
    """Match only equal-count measures with identical pitch/rest/chord structure.

    Duration is deliberately excluded. No edit-distance best matching, reference
    prefix, or timing-head output can influence this conservative pairing.
    """
    a, b = measures(reference), measures(predicted)
    if len(a) != len(b):
        return {}, dict(
            reason="measure-count-mismatch", reference_measures=len(a), predicted_measures=len(b)
        )
    pairs, rejected = {}, []
    for measure, (left, right) in enumerate(zip(a, b, strict=True), 1):
        if [key for _, key in left] != [key for _, key in right]:
            rejected.append(measure)
            continue
        pairs.update((i, j) for (i, _), (j, _) in zip(left, right, strict=True))
    return pairs, dict(reason="strict-measure-pairing", rejected_crop_relative_measures=rejected)


def legacy_timing(symbol):
    rhythm = symbol["rhythm"]
    if not rhythm.startswith(("note_", "rest_")) or "G" in rhythm or rhythm.endswith("m"):
        return {}
    kern = rhythm.split("_", 1)[1]
    duration = kern_to_symbol_duration(kern)
    types = {value: name for name, value in TYPE_QUARTERS.items()}
    values = {
        "timing.elapsed": str(duration.fraction * 4),
        "timing.ratio": f"{duration.actual_notes}:{duration.normal_notes}",
    }
    written = types.get(duration.base_duration * 4)
    if written:
        values["timing.written"] = f"{written}:{duration.dots}"
    if rhythm.startswith("rest_"):
        values["timing.full_rest"] = "yes" if kern == "0" else "no"
    return values


def prepare(args):
    from homr.transformer.vocabulary import EncodedSymbol
    from training.architecture.transformer.exact_timing_targets import build_timing_targets
    from training.omr_datasets.exact_timing_sidecar import attach_exact_timing_sidecar
    from training.transformer.training_vocabulary import read_tokens

    dataset = json.loads(args.dataset.read_text())
    rows = [row for row in dataset["rows"] if row["split"] == "valid"]
    if {r["score"] for r in rows} & {r["score"] for r in dataset["rows"] if r["split"] == "train"}:
        raise ValueError("Timing train/validation score overlap")
    rows.sort(key=lambda r: hashlib.sha256(("20261007:" + r["id"]).encode()).hexdigest())
    pilot, seen = [], set()
    for row in rows:
        key = row["source"], row["score"]
        if row["source"] == "ossq" or key not in seen:
            pilot.append(row["id"])
            seen.add(key)
    inference, evaluation = [], []
    for row in rows:
        for key, path in [
            ("image_sha256", row["image"]),
            ("label_sha256", row["label"]),
            ("timing_sha256", row["label"] + ".timing.json"),
        ]:
            if sha(path) != row[key]:
                raise ValueError("Qualified data changed: " + path)
        symbols = read_tokens(row["label"])
        attach_exact_timing_sidecar(row["label"], symbols)
        targets, diagnostics = build_timing_targets(
            [[EncodedSymbol("BOS"), *symbols, EncodedSymbol("EOS")]],
            dataset["target_manifest"]["classes"],
            duration_policy=dataset["target_manifest"]["durationPolicy"],
        )
        values = [
            {
                h: dataset["target_manifest"]["classes"][h][int(targets[h][0, i + 1])]
                for h in HEADS
                if int(targets[h][0, i + 1]) >= 0
            }
            for i in range(len(symbols))
        ]
        safe = {k: row[k] for k in ("id", "source", "score", "image", "image_sha256")}
        inference.append(safe)
        evaluation.append(
            dict(
                **safe,
                reference=[serialize(s) for s in symbols],
                targets=values,
                label=row["label"],
                label_sha256=row["label_sha256"],
                timing_sha256=row["timing_sha256"],
                diagnostics=diagnostics,
            )
        )
    models = dict(
        core=dataset["base_checkpoint"],
        core_sha256=dataset["base_sha256"],
        target_manifest=dataset["target_manifest"],
        heads={},
    )
    for epoch in (4, 6):
        path = args.dataset.parent / f"full/epoch-{epoch}.pth"
        models["heads"][f"epoch{epoch}"] = dict(path=str(path), sha256=sha(path))
    if sha(models["core"]) != models["core_sha256"]:
        raise ValueError("Core changed")
    args.run.mkdir(parents=True, exist_ok=False)
    write(args.run / "inference-full.json", inference)
    write(args.run / "inference-pilot.json", [r for r in inference if r["id"] in pilot])
    write(args.run / "evaluation.json", evaluation)
    write(args.run / "models.json", models)
    write(
        args.run / "preparation.json",
        dict(
            dataset_sha256=sha(args.dataset),
            code_sha256=sha(__file__),
            manifests_sha256={
                name: sha(args.run / name)
                for name in (
                    "inference-full.json",
                    "inference-pilot.json",
                    "evaluation.json",
                    "models.json",
                )
            },
            validation_crops=len(inference),
            pilot_crops=len(pilot),
            references_enter_inference=False,
            scope="Previously used validation, not untouched test. Raw eager crop greedy decoding, no production policies.",
        ),
    )
    print(f"Prepared {len(inference)} validation crops; {len(pilot)} pilot crops", flush=True)


def decode(args):
    import cv2
    import torch

    from homr.transformer.configs import Config
    from training.architecture.transformer.exact_timing_heads import load_timing_bundle
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.qualify_joint_holdout import pixels

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    manifest = args.run / f"inference-{args.scope}.json"
    preparation = json.loads((args.run / "preparation.json").read_text())
    for path in (manifest, args.run / "models.json"):
        if sha(path) != preparation["manifests_sha256"][path.name]:
            raise ValueError("Frozen inference manifest/models changed")
    rows = json.loads(manifest.read_text())[args.part :: args.parts]
    models = json.loads((args.run / "models.json").read_text())
    if sha(models["core"]) != models["core_sha256"]:
        raise ValueError("Core changed")
    folder = args.run / args.scope / f"part-{args.part:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    identity = dict(
        inference_manifest_sha256=sha(manifest),
        models_sha256=sha(args.run / "models.json"),
        code_sha256=sha(__file__),
        threads=args.threads,
        torch=str(torch.__version__),
        part=args.part,
        parts=args.parts,
        protocol="CPU float32 eager greedy, math SDPA, final-divider enforcement off",
        source_sha256={
            name: sha(repository / name)
            for name in (
                "homr/transformer/configs.py",
                "homr/transformer/vocabulary.py",
                "homr/staff_parsing.py",
                "training/architecture/transformer/tromr_arch.py",
                "training/architecture/transformer/encoder.py",
                "training/architecture/transformer/decoder.py",
                "training/architecture/transformer/custom_x_transformer.py",
                "training/architecture/transformer/exact_timing_heads.py",
                "training/transformer/qualify_joint_holdout.py",
            )
        },
    )
    identity_path = folder / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Resume identity changed")
    write(identity_path, identity)
    cfg = Config()
    cfg.encoder_pretrained = False
    cfg.enable_structured_heads = False
    cfg.enforce_final_divider = False
    model = TrOMR(cfg).eval()
    model.load_state_dict(
        torch.load(models["core"], map_location="cpu", weights_only=True), strict=True
    )
    model.requires_grad_(False)
    model.decoder.device = torch.device("cpu")
    heads = {}
    for arm, spec in models["heads"].items():
        if sha(spec["path"]) != spec["sha256"]:
            raise ValueError("Probe bundle changed")
        bundle = torch.load(spec["path"], map_location="cpu", weights_only=True)
        if bundle["trainingStatus"] != "probe-trained":
            raise ValueError("Require an explicitly research-only probe bundle")
        # Explicit research use. Never relabels a bundle as inference-qualified.
        heads[arm] = load_timing_bundle(
            bundle, models["target_manifest"], models["core_sha256"], allow_untrained=True
        ).eval()
        heads[arm].requires_grad_(False)
    captured = []

    def capture(module, inputs, output):
        captured.append((output[6][0, -1].clone(), int(output[0][0, -1].argmax())))

    with torch.inference_mode():
        for index, row in enumerate(rows):
            destination = folder / (hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".json")
            if destination.exists():
                previous = json.loads(destination.read_text())
                if previous["input"] != row or previous["identity"] != identity:
                    raise ValueError("Cached decoding differs")
                continue
            if sha(row["image"]) != row["image_sha256"]:
                raise ValueError("Image changed")
            tick = time.monotonic()
            captured.clear()
            hook = model.decoder.net.register_forward_hook(capture)
            try:
                predicted = model.generate(pixels(row["image"]).unsqueeze(0))
            finally:
                hook.remove()
            if len(captured) not in (len(predicted), len(predicted) + 1):
                raise ValueError("Generation hidden/token alignment changed")
            if len(captured) > len(predicted) and captured[-1][1] != cfg.eos_token:
                raise ValueError("Extra hidden step was not EOS")
            parity = None
            if args.scope == "pilot":
                plain = model.generate(pixels(row["image"]).unsqueeze(0))
                parity = [serialize(s) for s in predicted] == [serialize(s) for s in plain]
                if not parity:
                    raise ValueError("Read-only hidden capture changed recognition")
            hidden = (
                torch.stack([value for value, _ in captured[: len(predicted)]])
                if predicted
                else None
            )
            if hidden is not None and not torch.isfinite(hidden).all():
                raise ValueError("Nonfinite generated hidden states")
            timing = {}
            for arm, addon in heads.items():
                predictions = (
                    {h: logits.argmax(-1).tolist() for h, logits in addon(hidden).items()}
                    if hidden is not None
                    else {h: [] for h in HEADS}
                )
                timing[arm] = [
                    {h: models["target_manifest"]["classes"][h][predictions[h][i]] for h in HEADS}
                    for i in range(len(predicted))
                ]
            write(
                destination,
                dict(
                    input=row,
                    identity=identity,
                    prediction=[serialize(s) for s in predicted],
                    timing=timing,
                    seconds=time.monotonic() - tick,
                    reference_used=False,
                    capture_recognition_parity=parity,
                    generation_steps=len(captured),
                    sequence_limit_reached=len(captured) == cfg.max_seq_len
                    and len(predicted) == cfg.max_seq_len,
                ),
            )
            print(
                json.dumps(
                    dict(
                        id=row["id"],
                        completed=index + 1,
                        total=len(rows),
                        seconds=time.monotonic() - tick,
                    )
                ),
                flush=True,
            )
    write(folder / "completion.json", dict(success=True, rows=len(rows), identity=identity))


def score(args):
    from rapidfuzz.distance import Levenshtein

    preparation = json.loads((args.run / "preparation.json").read_text())
    for name in ("evaluation.json", f"inference-{args.scope}.json"):
        if sha(args.run / name) != preparation["manifests_sha256"][name]:
            raise ValueError("Frozen evaluation/inference manifest changed")

    evaluations = {row["id"]: row for row in json.loads((args.run / "evaluation.json").read_text())}
    expected = {
        row["id"] for row in json.loads((args.run / f"inference-{args.scope}.json").read_text())
    }
    cases, totals, recognition = (
        [],
        defaultdict(lambda: defaultdict(Counter)),
        defaultdict(lambda: defaultdict(Counter)),
    )
    paths = sorted((args.run / args.scope).glob("part-*/*.json"))
    records = [
        json.loads(p.read_text())
        for p in paths
        if p.name not in ("identity.json", "completion.json")
    ]
    if Counter(r["input"]["id"] for r in records) != Counter({rid: 1 for rid in expected}):
        raise ValueError("Require exactly one completed decoding for each frozen crop")
    for record in records:
        reference = evaluations[record["input"]["id"]]
        if record["reference_used"] or record["input"]["image_sha256"] != reference["image_sha256"]:
            raise ValueError("Reference-dependent or changed input")
        predicted = record["prediction"]
        pairs, alignment = match_events(reference["reference"], predicted)
        source = reference["source"]
        for metric, projection in [
            ("symbol", lambda x: (x["rhythm"], x["pitch"], x["lift"])),
            ("rhythm", lambda x: x["rhythm"]),
            ("pitch", lambda x: x["pitch"]),
            ("accidental", lambda x: x["lift"]),
        ]:
            note_only = metric in ("pitch", "accidental")

            def project(sequence):
                return [
                    projection(s)
                    for s in sequence
                    if s["rhythm"].startswith("note" if note_only else ("note", "rest"))
                ]

            left, right = project(reference["reference"]), project(predicted)
            recognition[source][metric].update(
                errors=Levenshtein.distance(left, right), support=len(left)
            )
        observations = []
        for ref_index, targets in enumerate(reference["targets"]):
            if not targets:
                continue
            pred_index = pairs.get(ref_index)
            legacy = legacy_timing(predicted[pred_index]) if pred_index is not None else {}
            for head, target in targets.items():
                for arm in ("legacy", *record["timing"]):
                    value = (
                        legacy.get(head)
                        if arm == "legacy"
                        else (
                            record["timing"][arm][pred_index][head]
                            if pred_index is not None
                            else None
                        )
                    )
                    counts = totals[source][arm + "/" + head]
                    counts.update(
                        support=1, matched=int(pred_index is not None), correct=int(value == target)
                    )
                    if pred_index is not None:
                        counts.update(matched_correct=int(value == target))
                    observations.append(
                        dict(
                            reference_token_index=ref_index,
                            predicted_token_index=pred_index,
                            head=head,
                            arm=arm,
                            target=target,
                            prediction=value,
                        )
                    )
        cases.append(
            dict(
                id=reference["id"],
                source=source,
                score=reference["score"],
                alignment=alignment,
                reference_used_in_decoding=False,
                timing=observations,
            )
        )
    report = dict(
        cases=cases,
        recognition=recognition,
        timing=totals,
        evaluation_sha256=sha(args.run / "evaluation.json"),
        code_sha256=sha(__file__),
        prediction_files_sha256={str(p): sha(p) for p in paths},
        limitation="Raw greedy crop research probe, not the accepted full-policy export. References only enter scoring. "
        "Strict equal-measure/pitch/rest/chord matching; unmatched reference targets retained in denominator. "
        "Timing heads are passive and cannot change recognition. Previously used validation, not untouched test. "
        "No runtime activation, tuplet-bracket preservation or full-page serialization claim.",
    )
    write(args.run / f"{args.scope}-report.json", report)
    print(json.dumps(dict(recognition=recognition, timing=totals), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--run", type=Path, required=True)
    decode_parser = sub.add_parser("decode")
    decode_parser.add_argument("--run", type=Path, required=True)
    decode_parser.add_argument("--scope", choices=["pilot", "full"], required=True)
    decode_parser.add_argument("--threads", type=int, default=2)
    decode_parser.add_argument("--part", type=int, default=0)
    decode_parser.add_argument("--parts", type=int, default=1)
    score_parser = sub.add_parser("score")
    score_parser.add_argument("--run", type=Path, required=True)
    score_parser.add_argument("--scope", choices=["pilot", "full"], required=True)
    args = parser.parse_args()
    if args.phase == "decode" and not (0 <= args.part < args.parts and args.threads > 0):
        parser.error("Invalid worker partition/threads")
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
