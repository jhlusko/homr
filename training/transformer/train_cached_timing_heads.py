"""Resumable frozen-core timing probe on hash-bound, teacher-forced features.

No core weights move. Teacher-forced validation is not free-decoding qualification.
"""

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from training.architecture.transformer.exact_timing_heads import (
    contract_digest,
    initialize_timing_bundle,
    load_timing_bundle,
    timing_loss,
)
from training.architecture.transformer.exact_timing_targets import HEADS, build_timing_targets
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.omr_datasets.exact_timing_sidecar import attach_exact_timing_sidecar
from training.onnx.qualify_joint_export import sha, write
from training.transformer.training_vocabulary import read_tokens, to_decoder_branches

SEED = 20261006


def atomic_save(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def freeze_identity(path, identity):
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Resume identity changed: use a new output directory")
    write(path, identity)


def source_balanced_schedule(rows, epoch, crops_per_source, seed=SEED):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] == "train":
            groups[row["source"]].append(index)
    if set(groups) != {"ossq", "lieder"} or crops_per_source <= 0:
        raise ValueError("Requires both training sources and positive batch size")
    rng = random.Random(seed + epoch)
    length = max(map(len, groups.values()))
    padded = {}
    for source, indices in sorted(groups.items()):
        values = []
        while len(values) < length:
            cycle = indices.copy()
            rng.shuffle(cycle)
            values.extend(cycle)
        padded[source] = values[:length]
    return [
        [
            index
            for source in sorted(padded)
            for index in padded[source][start : start + crops_per_source]
        ]
        for start in range(0, length, crops_per_source)
    ]


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    lieder = json.loads(args.lieder_report.read_text())
    ossq = json.loads(args.ossq_packet.read_text())
    valid_ossq = json.loads(args.ossq_valid.read_text())
    manifest = json.loads(args.target_manifest.read_text())
    if ossq["target_manifest_sha256"] != sha(args.target_manifest) or valid_ossq[
        "target_manifest_sha256"
    ] != sha(args.target_manifest):
        raise ValueError("Timing manifests differ")
    models = json.loads(args.models.read_text())["accepted_oldcore_v7"]
    core = Path(models["core"])
    if sha(core) != models["sha256"]["core"]:
        raise ValueError("Accepted core hash changed")
    rows = []

    def add(source, split, score, image, label, expected_image_hash):
        image, label = Path(image), Path(label)
        if sha(image) != expected_image_hash:
            raise ValueError("Qualified image changed")
        symbols = read_tokens(label)
        attach_exact_timing_sidecar(label, symbols)
        if len(symbols) + 2 > 608:
            raise ValueError("Qualified crop exceeds decoder context")
        to_decoder_branches(symbols)
        rows.append(
            dict(
                id=source + "/" + label.name,
                source=source,
                split=split,
                score=score,
                image=str(image),
                label=str(label),
                image_sha256=expected_image_hash,
                label_sha256=sha(label),
                timing_sha256=sha(Path(str(label) + ".timing.json")),
                symbols=len(symbols),
            )
        )

    for score in lieder["scores"]:
        for crop in score["crops"]:
            if not crop["exact_legacy_tokens"]:
                continue
            if len(crop["splits"]) != 1:
                raise ValueError("Ambiguous Lieder split")
            label = Path(crop["replayed_label"])
            if sha(label) != sha(Path(crop["token"])):
                raise ValueError("Lieder no longer matches public tokens")
            add(
                "lieder",
                crop["splits"][0],
                score["score"],
                crop["image"],
                label,
                crop["image_sha256"],
            )
    for record in ossq["records"]:
        label = args.ossq_packet.parent / "crops" / (record["stem"] + ".txt")
        if (
            sha(label) != record["token_sha256"]
            or sha(Path(str(label) + ".timing.json")) != record["timing_sha256"]
        ):
            raise ValueError("OSSQ packet changed")
        add(
            "ossq",
            "train",
            record["stem"].split("_")[0],
            record["image"],
            label,
            record["image_sha256"],
        )
    for record in valid_ossq["records"]:
        if record["split"] != "valid":
            continue
        label = args.ossq_valid.parent / (record["stem"] + ".txt")
        repair = json.loads((args.repaired_overlay / "receipt.json").read_text())
        public = Path(next(r["public"] for r in repair["records"] if r["stem"] == record["stem"]))
        images = list(public.parent.glob(record["stem"] + "-*.png"))
        if len(images) != 1:
            raise ValueError("Ambiguous corrected validation image")
        if (
            sha(label) != record["token_sha256"]
            or sha(Path(str(label) + ".timing.json")) != record["timing_sha256"]
        ):
            raise ValueError("Corrected OSSQ validation changed")
        add("ossq", "valid", record["stem"].split("_")[0], images[0], label, sha(images[0]))
    train_scores = {r["score"] for r in rows if r["split"] == "train"}
    valid_scores = {r["score"] for r in rows if r["split"] == "valid"}
    if train_scores & valid_scores or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate crops or score split overlap")
    # Interleave source/split strata so a bounded extraction benchmark samples both.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["split"])].append(row)
    for values in groups.values():
        values.sort(key=lambda r: hashlib.sha256((str(SEED) + r["id"]).encode()).hexdigest())
    ordered = [
        groups[key][index]
        for index in range(max(map(len, groups.values())))
        for key in sorted(groups)
        if index < len(groups[key])
    ]
    write(
        args.out / "dataset.json",
        dict(
            schema="homr.timing-probe-dataset.v1",
            rows=ordered,
            base_checkpoint=str(core),
            base_sha256=sha(core),
            target_manifest=manifest,
            input_sha256={
                str(p): sha(p)
                for p in (
                    args.lieder_report,
                    args.ossq_packet,
                    args.ossq_valid,
                    args.repaired_overlay / "receipt.json",
                    args.models,
                    args.target_manifest,
                )
            },
            counts={str(k): len(v) for k, v in groups.items()},
            limitation="Teacher-forced probe. OSSQ valid: only eight corrected crops from two scores; "
            "Lieder valid: 291 unchanged crops. Common-class support, not release qualification.",
        ),
    )
    print(json.dumps({str(k): len(v) for k, v in groups.items()}), flush=True)


def extract(args):
    from homr.transformer.configs import Config
    from homr.transformer.vocabulary import EncodedSymbol
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.transformer.qualify_joint_holdout import pixels

    dataset = json.loads(args.dataset.read_text())
    if sha(Path(dataset["base_checkpoint"])) != dataset["base_sha256"]:
        raise ValueError("Frozen core changed")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    args.out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        dataset_sha256=sha(args.dataset),
        base_sha256=dataset["base_sha256"],
        target_contract=contract_digest(dataset["target_manifest"]),
        threads=args.threads,
        torch_version=str(torch.__version__),
        code_sha256=sha(Path(__file__)),
        protocol="eval, float32 CPU, teacher forcing; hidden[t] predicts token[t+1]",
        source_sha256={
            name: sha(Path(__file__).resolve().parents[2] / name)
            for name in (
                "homr/transformer/configs.py",
                "homr/transformer/vocabulary.py",
                "homr/staff_parsing.py",
                "training/architecture/transformer/tromr_arch.py",
                "training/architecture/transformer/encoder.py",
                "training/architecture/transformer/decoder.py",
                "training/architecture/transformer/custom_x_transformer.py",
                "training/architecture/transformer/exact_timing_targets.py",
            )
        },
    )
    freeze_identity(args.out / "identity.json", identity)
    cfg = Config()
    cfg.encoder_pretrained = False
    cfg.enable_structured_heads = False
    cfg.enforce_final_divider = False
    model = TrOMR(cfg).eval()
    model.load_state_dict(
        torch.load(dataset["base_checkpoint"], map_location="cpu", weights_only=True), strict=True
    )
    model.requires_grad_(False)
    model.decoder.device = torch.device("cpu")
    selected = dataset["rows"][: args.limit] if args.limit else dataset["rows"]
    results = []
    with torch.inference_mode():
        for row in selected:
            name = hashlib.sha256(row["id"].encode()).hexdigest()[:24] + ".pt"
            path = args.out / name
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if payload["row"] != row or payload["identity"] != identity:
                    raise ValueError("Feature resume identity mismatch")
            else:
                tick = time.monotonic()
                for key, p in (
                    ("image_sha256", row["image"]),
                    ("label_sha256", row["label"]),
                    ("timing_sha256", row["label"] + ".timing.json"),
                ):
                    if sha(Path(p)) != row[key]:
                        raise ValueError("Qualified extraction input changed")
                symbols = read_tokens(row["label"])
                attach_exact_timing_sidecar(row["label"], symbols)
                branches = to_decoder_branches(symbols)
                n = len(symbols) + 2
                inputs = {key: value[:n].unsqueeze(0) for key, value in vars(branches).items()}
                hidden = model(inputs=pixels(row["image"]).unsqueeze(0), **inputs)["hidden"][0]
                padded = [EncodedSymbol("BOS"), *symbols, EncodedSymbol("EOS")]
                targets, diagnostics = build_timing_targets(
                    [padded],
                    dataset["target_manifest"]["classes"],
                    duration_policy=dataset["target_manifest"]["durationPolicy"],
                )
                targets = {h: value[0, 1:] for h, value in targets.items()}
                if any(len(value) != len(hidden) for value in targets.values()):
                    raise ValueError("Decoder target shift mismatch")
                keep = torch.stack([value != IGNORE_INDEX for value in targets.values()]).any(0)
                payload = dict(
                    row=row,
                    identity=identity,
                    hidden=hidden[keep].clone(),
                    targets={h: v[keep].clone() for h, v in targets.items()},
                    token_indices=torch.where(keep)[0],
                    diagnostics=diagnostics,
                    seconds=time.monotonic() - tick,
                )
                atomic_save(path, payload)
            results.append(
                dict(
                    path=str(path),
                    sha256=sha(path),
                    row=row,
                    seconds=payload["seconds"],
                    events=len(payload["hidden"]),
                )
            )
            write(
                args.out / "features.json",
                dict(
                    identity=identity,
                    records=results,
                    complete=len(results) == len(dataset["rows"]),
                    total_crops=len(dataset["rows"]),
                ),
            )
            print(
                f"features {len(results)}/{len(selected)} {row['source']}/{row['split']} {payload['seconds']:.3f}s",
                flush=True,
            )
    if sha(Path(dataset["base_checkpoint"])) != dataset["base_sha256"]:
        raise ValueError("Frozen core changed during extraction")


def load_features(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise ValueError("Cached features changed")
    return torch.load(path, map_location="cpu", weights_only=True)


def evaluate(heads, records, classes):
    metrics = {}
    heads.eval()
    with torch.no_grad():
        for source in ("ossq", "lieder"):
            support = {h: Counter() for h in HEADS}
            correct = {h: Counter() for h in HEADS}
            losses = {h: 0.0 for h in HEADS}
            for record in records:
                if record["row"]["split"] != "valid" or record["row"]["source"] != source:
                    continue
                payload = load_features(record)
                logits = heads(payload["hidden"])
                loss = timing_loss(logits, payload["targets"], weights={h: 1.0 for h in HEADS})
                for item in loss.heads:
                    h = item.name
                    losses[h] += float(item.loss) * item.supervised
                    predicted, target = logits[h].argmax(-1), payload["targets"][h]
                    for index, name in enumerate(classes[h]):
                        mask = target == index
                        support[h][name] += int(mask.sum())
                        correct[h][name] += int(((predicted == target) & mask).sum())
            metrics[source] = {
                h: dict(
                    support=dict(support[h]),
                    correct=dict(correct[h]),
                    eligible=sum(support[h].values()),
                    loss=losses[h] / max(1, sum(support[h].values())),
                    accuracy=sum(correct[h].values()) / max(1, sum(support[h].values())),
                )
                for h in HEADS
            }
    return metrics


def train(args):
    torch.set_num_threads(args.threads)
    features = json.loads(args.features.read_text())
    dataset = json.loads(args.dataset.read_text())
    if features["identity"]["dataset_sha256"] != sha(args.dataset):
        raise ValueError("Features belong to another dataset")
    if not features["complete"] and not args.allow_partial:
        raise ValueError("Incomplete features: explicit --allow-partial required for smoke only")
    records = features["records"]
    rows = [r["row"] for r in records]
    for source in ("ossq", "lieder"):
        if not any(r["source"] == source and r["split"] == "valid" for r in rows):
            raise ValueError("Both source validation sets required")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = dataset["target_manifest"]
    identity = dict(
        feature_manifest_sha256=sha(args.features),
        dataset_sha256=sha(args.dataset),
        lr=args.lr,
        crops_per_source=args.crops_per_source,
        seed=SEED,
        threads=args.threads,
        weights={h: 1.0 for h in HEADS},
        code_sha256=sha(Path(__file__)),
        scope="partial-cache smoke" if not features["complete"] else "complete-cache probe",
        sampling="Equal source crop draws; per-source per-head mean CE, then equal source average",
    )
    freeze_identity(args.out / "identity.json", identity)
    latest = args.out / "latest.pth"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Training resume identity mismatch")
        bundle = state["bundle"]
    else:
        dim = load_features(records[0])["hidden"].shape[-1]
        bundle = initialize_timing_bundle(dim, manifest, dataset["base_sha256"], seed=SEED)
        state = dict(epoch=0, next_batch=0, history=[], identity=identity, bundle=bundle)
    heads = load_timing_bundle(bundle, manifest, dataset["base_sha256"], allow_untrained=True)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr)
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    else:
        write(args.out / "initial-validation.json", evaluate(heads, records, manifest["classes"]))
    updates = 0
    for epoch in range(state["epoch"], args.epochs):
        schedule = source_balanced_schedule(rows, epoch, args.crops_per_source)
        start = state["next_batch"] if epoch == state["epoch"] else 0
        for batch_index in range(start, len(schedule)):
            heads.train()
            optimizer.zero_grad(set_to_none=True)
            source_losses = []
            batch_support = Counter()
            running = state.setdefault("running", {})
            for source in ("ossq", "lieder"):
                payloads = [
                    load_features(records[index])
                    for index in schedule[batch_index]
                    if rows[index]["source"] == source
                ]
                hidden = torch.cat([p["hidden"] for p in payloads])
                targets = {h: torch.cat([p["targets"][h] for p in payloads]) for h in HEADS}
                measured = timing_loss(heads(hidden), targets, weights=identity["weights"])
                source_losses.append(measured.total)
                for item in measured.heads:
                    batch_support[item.name] += item.supervised
                    key = source + "/" + item.name
                    totals = running.setdefault(key, dict(loss_sum=0.0, support=0))
                    totals["loss_sum"] += float(item.loss.detach()) * item.supervised
                    totals["support"] += item.supervised
            loss = torch.stack(source_losses).mean()
            loss.backward()
            for head in HEADS:
                for parameter in heads.projections[head.replace(".", "_")].parameters():
                    if parameter.grad is None:
                        if batch_support[head]:
                            raise ValueError("Missing supervised timing-head gradients: " + head)
                    elif not torch.isfinite(parameter.grad).all():
                        raise ValueError("Nonfinite timing-head gradients: " + head)
            optimizer.step()
            bundle.update(heads=heads.state_dict(), trainingStatus="probe-trained")
            state.update(
                epoch=epoch,
                next_batch=batch_index + 1,
                bundle=bundle,
                optimizer=optimizer.state_dict(),
                last_loss=float(loss.detach()),
            )
            atomic_save(latest, state)
            updates += 1
            if args.max_updates and updates >= args.max_updates:
                print("Stopped at resumable update limit", flush=True)
                return
        metrics = evaluate(heads, records, manifest["classes"])
        state["history"].append(
            dict(
                epoch=epoch + 1,
                validation=metrics,
                last_train_batch_loss=state["last_loss"],
                train={
                    key: dict(
                        loss=value["loss_sum"] / max(1, value["support"]), support=value["support"]
                    )
                    for key, value in state["running"].items()
                },
            )
        )
        state.update(epoch=epoch + 1, next_batch=0)
        state.pop("running", None)
        atomic_save(latest, state)
        atomic_save(args.out / f"epoch-{epoch + 1}.pth", bundle)
        write(args.out / "history.json", state["history"])
        print(
            f"epoch {epoch + 1} validation "
            + json.dumps({s: {h: m["accuracy"] for h, m in v.items()} for s, v in metrics.items()}),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    p = sub.add_parser("prepare")
    for name in (
        "lieder-report",
        "ossq-packet",
        "ossq-valid",
        "repaired-overlay",
        "models",
        "target-manifest",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("extract")
    for name in ("dataset", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("train")
    for name in ("dataset", "features", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--crops-per-source", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--max-updates", type=int, default=0)
    args = parser.parse_args()
    globals()[args.phase](args)


if __name__ == "__main__":
    main()
