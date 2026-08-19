"""
Train the Dynamic-mark classifier, and report accuracy per label - not just overall.

`dynamics_crops.py` found 18 raw labels dominated by 8 that cover ~97% of examples; an
overall accuracy number would be dominated by "p" and "f" the same way 27.49's tie head
and 27.62's OCR sweep were both dominated by their majority class. Per-label accuracy is
what says whether the tail (fff, sfz, rf, ...) is learned at all or silently ignored.
"""

# flake8: noqa: T201

import argparse
import collections
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.architecture.ocr.dynamics_classifier import DynamicsCNN, Labels
from training.ocr.dynamics_data import DynamicsCrops, collate, labels_of, read_manifest


def evaluate(model: DynamicsCNN, loader: DataLoader, labels: Labels, device: str) -> dict:
    model.eval()
    correct = collections.Counter()
    total = collections.Counter()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            predicted = logits.argmax(dim=-1).cpu()
            for pred, truth in zip(predicted.tolist(), batch["targets"].tolist(), strict=True):
                label = labels.decode(truth)
                total[label] += 1
                correct[label] += int(pred == truth)
    return {"correct": correct, "total": total}


def describe(result: dict) -> str:
    total = sum(result["total"].values())
    correct = sum(result["correct"].values())
    lines = [f"overall: {correct}/{total} ({correct / max(1, total):.1%})"]
    for label, count in result["total"].most_common():
        acc = result["correct"][label] / count
        lines.append(f"  {label:<10} {result['correct'][label]:>4}/{count:<4} ({acc:.1%})")
    return "\n".join(lines)


def train(args: argparse.Namespace) -> dict:
    train_samples = read_manifest(args.train)
    valid_samples = read_manifest(args.valid)
    labels = labels_of(train_samples)

    model = DynamicsCNN(len(labels)).to(args.device)
    train_set = DynamicsCrops(train_samples, labels)
    valid_set = DynamicsCrops(valid_samples, labels)
    print(f"train {len(train_set):,} crops, valid {len(valid_set):,}")
    print(f"{len(labels)} labels")
    if train_set.skipped or valid_set.skipped:
        print(f"  {train_set.skipped + valid_set.skipped} skipped: label absent from training")

    loaders = {
        "train": DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate,
        ),
        "valid": DataLoader(
            valid_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate
        ),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for batch in loaders["train"]:
            logits = model(batch["images"].to(args.device))
            loss = loss_fn(logits, batch["targets"].to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        result = evaluate(model, loaders["valid"], labels, args.device)
        valid_total = sum(result["total"].values())
        valid_correct = sum(result["correct"].values())
        print(
            f"epoch {epoch}: loss {total_loss / max(1, batches):.4f}  "
            f"valid {valid_correct}/{valid_total} ({valid_correct / max(1, valid_total):.1%})"
        )
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(1, batches),
                "valid_exact": valid_correct / max(1, valid_total),
            }
        )

    print(describe(result))
    if args.weights:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "labels": labels.labels}, args.weights)
    return {"history": history, "labels": labels.labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--train", type=Path, required=True, help="train.jsonl from dynamics_crops.")
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = train(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
