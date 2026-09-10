"""Score saved detector checkpoints against one another on identical validation data.

The experiment matrix compares E0 (synthetic only) with E1-E3 (synthetic plus real
scans), and the obvious way to read the result - each run's own end-of-training
`valid:` block - does not actually compare them. E0 validated on the synthetic bank;
E1-E3 validated on a mixed bank that also contains real-scan patches. Different
measurement, different difficulty, no shared baseline: whichever way those numbers had
fallen, they could not have answered whether scan data helped.

This scores any checkpoint against any bank, so every model can be put on the same
validation set - and on each set separately, which is the more informative view:
"did adding scans cost anything on synthetic pages" and "did adding scans help on real
ones" are two questions, and a single mixed number silently averages them.

`ignore_index` is passed through by default, because a scan-derived mask marks
unsupervised pixels with a sentinel rather than a class. Scoring those pixels counts
every ignored region as a prediction error - it is what made E1-E3's first validation
numbers unusable while their training IoU (which did pass the argument) looked normal.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.ocr.detector_patches import PreExtractedPatches, read_index
from training.ocr.train_detector import (
    NUM_CLASSES,
    CamVidModel,
    collate,
    evaluate,
)


def load_checkpoint(path: Path, device: str) -> CamVidModel:
    model = CamVidModel(
        arch="Unet", encoder_name="resnet18", in_channels=3, out_classes=NUM_CLASSES,
        skip_weights_download=True,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def score(
    checkpoint: Path, index: Path, device: str = "cpu", ignore_index: int | None = 255,
    batch_size: int = 16, workers: int = 4,
) -> dict[str, float]:
    dataset = PreExtractedPatches(read_index(index))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate,
    )
    return evaluate(load_checkpoint(checkpoint, device), loader, device, ignore_index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, required=True, nargs="+",
        help="One or more saved .pth files, scored in turn.",
    )
    parser.add_argument(
        "--index", type=Path, required=True, nargs="+",
        help="One or more pre-extracted bank indexes, each scored separately.",
    )
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    results: dict[str, dict[str, dict[str, float]]] = {}
    for index in args.index:
        # Banks are laid out as <bank>/<split>/index.txt, so the split alone ("valid")
        # is not a name - every bank has one, and keying results by it silently
        # overwrites one evaluation with another.
        set_name = (
            f"{index.parent.parent.name}/{index.parent.name}"
            if index.name == "index.txt"
            else index.stem
        )
        results[set_name] = {}
        for checkpoint in args.checkpoint:
            per_class = score(
                checkpoint, index, args.device, args.ignore_index,
                args.batch_size, args.workers,
            )
            results[set_name][checkpoint.stem] = per_class
            print(f"{set_name} / {checkpoint.stem}")
            for name, value in per_class.items():
                print(f"  {name:<14} IoU {value:.3f}")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
