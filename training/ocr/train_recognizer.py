"""
Train the syllable recogniser, and report the number that can actually be wrong.

27.49 is the reason this file reports what it reports. A tie head with micro F1 0.997 looked
excellent while being near-useless, because one class carried the whole figure. The same
trap is waiting here in a different shape: **17.0% of validation syllables never appear in
training** (27.48), so a model that memorised the training vocabulary and read nothing would
still score 83% - and 83% reads like a working recogniser.

So accuracy is always split:

    seen      syllables whose exact string appears in training
    unseen    syllables that do not

The second number is the one that says whether the model reads. The first says whether it
remembers. A gap between them is the finding, and an aggregate hides it - which is 27.16's
crosstab-over-totals in a new place.

Character error rate is reported alongside, because exact-match alone cannot distinguish a
recogniser that is close from one that is lost, and for a lyric a near miss is recoverable
by a human where a wrong word is not.
"""

# flake8: noqa: T201

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from training.architecture.ocr.crnn import BLANK, IMAGE_HEIGHT, CRNN, Alphabet
from training.ocr.recognizer_data import (
    SyllableCrops,
    alphabet_of,
    collate,
    read_manifest,
)


def edit_distance(first: str, second: str) -> int:
    """Levenshtein, for character error rate."""
    if not first:
        return len(second)
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
            )
        previous = current
    return previous[-1]


@dataclass
class Accuracy:
    exact: int = 0
    total: int = 0
    distance: int = 0
    characters: int = 0
    examples: list[tuple[str, str]] = field(default_factory=list)

    def observe(self, truth: str, predicted: str) -> None:
        self.total += 1
        self.exact += truth == predicted
        self.distance += edit_distance(truth, predicted)
        self.characters += len(truth)
        if truth != predicted and len(self.examples) < 8:
            self.examples.append((truth, predicted))

    def describe(self, label: str) -> str:
        if not self.total:
            return f"{label}: nothing to score"
        rate = self.distance / max(1, self.characters)
        return (
            f"{label}: exact {self.exact / self.total:.1%} ({self.exact:,}/{self.total:,}), "
            f"CER {rate:.1%}"
        )


def evaluate(
    model: CRNN, loader: DataLoader, alphabet: Alphabet, seen: set[str], device: str
) -> tuple[Accuracy, Accuracy]:
    """Score the model, split by whether the syllable was ever in training."""
    model.eval()
    known, novel = Accuracy(), Accuracy()

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["images"].to(device))
            best = logits.argmax(dim=-1).permute(1, 0).cpu()
            for row, truth in zip(best, batch["texts"]):
                predicted = alphabet.decode(row.tolist())
                (known if truth in seen else novel).observe(truth, predicted)
    return known, novel


def train(args: argparse.Namespace) -> dict:
    train_samples = read_manifest(args.train)
    valid_samples = read_manifest(args.valid)

    # The alphabet comes from training only. A character that appears solely in validation
    # is one the model could never emit, and including it would quietly inflate the output
    # layer while pretending the character is learnable.
    alphabet = alphabet_of(train_samples)
    unrepresentable = {
        sample.text for sample in valid_samples if set(sample.text) - set(alphabet.characters)
    }

    model = CRNN(len(alphabet), image_height=args.height).to(args.device)
    train_set = SyllableCrops(train_samples, alphabet, model.frame_count, height=args.height)
    valid_set = SyllableCrops(
        [s for s in valid_samples if s.text not in unrepresentable],
        alphabet, model.frame_count, height=args.height,
    )

    print(f"train {len(train_set):,} crops, valid {len(valid_set):,}")
    print(f"alphabet {len(alphabet) - 1} characters plus blank")
    if train_set.too_long or valid_set.too_long:
        print(f"  {train_set.too_long + valid_set.too_long} refused: label longer than frames")
    if unrepresentable:
        print(f"  {len(unrepresentable)} valid syllables use characters absent from training")

    loaders = {
        "train": DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, collate_fn=collate,
        ),
        "valid": DataLoader(
            valid_set, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate
        ),
    }
    seen = {sample.text for sample in train_samples}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch in loaders["train"]:
            logits = model(batch["images"].to(args.device))
            input_lengths = torch.tensor(
                [model.frame_count(int(width)) for width in batch["widths"]]
            )
            loss = F.ctc_loss(
                logits,
                batch["targets"].to(args.device),
                input_lengths,
                batch["target_lengths"],
                blank=BLANK,
                zero_infinity=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1

        known, novel = evaluate(model, loaders["valid"], alphabet, seen, args.device)
        print(f"epoch {epoch}: loss {total / max(1, batches):.4f}")
        print(f"  {known.describe('seen  ')}")
        print(f"  {novel.describe('unseen')}")
        history.append(
            {
                "epoch": epoch,
                "loss": total / max(1, batches),
                "seen_exact": known.exact / max(1, known.total),
                "unseen_exact": novel.exact / max(1, novel.total),
                "seen_cer": known.distance / max(1, known.characters),
                "unseen_cer": novel.distance / max(1, novel.characters),
            }
        )

    if novel.examples:
        print("  unseen misreads: " + ", ".join(f"{t!r}->{p!r}" for t, p in novel.examples[:6]))
    if args.weights:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "alphabet": alphabet.characters}, args.weights)
    return {"history": history, "alphabet": alphabet.characters}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--train", type=Path, required=True, help="train.jsonl from lyric_crops.")
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path, help="Where to write the history as json.")
    parser.add_argument(
        "--height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Crops are normalised to this height. Must be a multiple of 16. 32 downscales "
        "93%% of crops, 48 downscales 43%%, 64 downscales 5%%.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
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
