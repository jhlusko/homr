"""
Where the recogniser's errors are, rather than how many there are.

27.54 killed the resolution hypothesis for the 11.6% CER and pointed at capacity or training
length. Both are expensive to test and neither is worth testing until the errors are known
to be spread rather than concentrated - a rate says how often, and 27.16's rule is that a
total can hide a structure that changes what to do about it.

So the same accuracy is cut several ways:

  * **by syllable length** - CTC needs a frame per character, and a long syllable in a narrow
    crop is squeezed. If errors climb with length, the frame budget is the constraint and
    the fix is geometric.
  * **by whether the label carries punctuation** - 14.6% of the corpus does, and a comma or
    semicolon is a few pixels of ink that a downscale can erase. If errors concentrate here,
    the fix is the crop margin.
  * **by whether the syllable was seen in training** - already reported per epoch, repeated
    here so the cuts can be read against it.
  * **by which characters are confused for which** - the one that names a cause rather than
    a correlate.

Runs on the CPU. A small CRNN over four thousand crops is a minute, and the GPU is usually
busy with the notation heads.
"""

# flake8: noqa: T201

import argparse
import collections
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.architecture.ocr.crnn import CRNN, Alphabet
from training.ocr.recognizer_data import SyllableCrops, collate, read_manifest
from training.ocr.train_recognizer import edit_distance


def load(weights: Path, height: int, device: str) -> tuple[CRNN, Alphabet]:
    saved = torch.load(weights, map_location=device, weights_only=False)
    alphabet = Alphabet(saved["alphabet"])
    model = CRNN(len(alphabet), image_height=height).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, alphabet


def predictions(model, alphabet, loader, device) -> list[tuple[str, str]]:
    """Every (truth, prediction) pair, each decoded over its own unpadded frames."""
    pairs = []
    with torch.no_grad():
        for batch in loader:
            best = model(batch["images"].to(device)).argmax(dim=-1).permute(1, 0).cpu()
            for row, truth, width in zip(best, batch["texts"], batch["widths"]):
                frames = model.frame_count(int(width))
                pairs.append((truth, alphabet.decode(row[:frames].tolist())))
    return pairs


def by_bucket(pairs: list[tuple[str, str]], bucket) -> dict:
    """Exact-match rate and CER within each bucket, plus its size."""
    grouped: dict = collections.defaultdict(lambda: [0, 0, 0, 0])
    for truth, predicted in pairs:
        row = grouped[bucket(truth)]
        row[0] += truth == predicted
        row[1] += 1
        row[2] += edit_distance(truth, predicted)
        row[3] += len(truth)
    return {
        key: (exact / max(1, total), distance / max(1, characters), total)
        for key, (exact, total, distance, characters) in sorted(grouped.items())
    }


def confusions(pairs: list[tuple[str, str]], limit: int = 12) -> list[tuple[str, int]]:
    """Which characters go missing or appear spuriously.

    Aligned crudely - a character present in the truth and absent from the prediction is
    counted as dropped. It does not distinguish a substitution from a deletion, which is
    enough to say *whether* one class of character is disappearing.
    """
    counter: collections.Counter[str] = collections.Counter()
    for truth, predicted in pairs:
        if truth == predicted:
            continue
        missing = collections.Counter(truth) - collections.Counter(predicted)
        extra = collections.Counter(predicted) - collections.Counter(truth)
        for character, count in missing.items():
            counter[f"dropped {character!r}"] += count
        for character, count in extra.items():
            counter[f"added {character!r}"] += count
    return counter.most_common(limit)


def report(pairs: list[tuple[str, str]], seen: set[str]) -> str:
    exact = sum(truth == predicted for truth, predicted in pairs)
    distance = sum(edit_distance(t, p) for t, p in pairs)
    characters = sum(len(t) for t, _ in pairs)

    lines = [
        f"{len(pairs):,} crops: exact {exact / len(pairs):.1%}, CER {distance / characters:.1%}",
        "",
        "by syllable length (does the frame budget bind?):",
    ]
    for length, (rate, cer, count) in by_bucket(pairs, len).items():
        lines.append(f"  {length:>2} chars: exact {rate:>6.1%}  CER {cer:>6.1%}  (n={count:,})")

    lines += ["", "by whether the label carries punctuation:"]
    marks = set(".,;:!?'’")
    for key, (rate, cer, count) in by_bucket(
        pairs, lambda t: "punctuation" if set(t) & marks else "none"
    ).items():
        lines.append(f"  {key:>12}: exact {rate:>6.1%}  CER {cer:>6.1%}  (n={count:,})")

    lines += ["", "by whether the syllable was seen in training:"]
    for key, (rate, cer, count) in by_bucket(
        pairs, lambda t: "seen" if t in seen else "unseen"
    ).items():
        lines.append(f"  {key:>12}: exact {rate:>6.1%}  CER {cer:>6.1%}  (n={count:,})")

    lines += ["", "characters dropped or added on wrong predictions:"]
    for label, count in confusions(pairs):
        lines.append(f"  {label}: {count:,}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True, help="For the seen/unseen split.")
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, alphabet = load(args.weights, args.height, args.device)
    known = set(alphabet.characters)
    samples = [s for s in read_manifest(args.valid) if not (set(s.text) - known)]
    loader = DataLoader(
        SyllableCrops(samples, alphabet, model.frame_count, height=args.height),
        batch_size=64, collate_fn=collate,
    )
    seen = {sample.text for sample in read_manifest(args.train)}
    print(report(predictions(model, alphabet, loader, args.device), seen))


if __name__ == "__main__":
    main()
