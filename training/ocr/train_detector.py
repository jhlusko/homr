"""
Train the text detector on the sampled patches - and measure per class before reaching for
any correction.

27.68 measured the detector's classes at up to 1,716x apart, more than three orders of
magnitude worse than the tie head's imbalance. 27.72 is the reason this file does not treat
that measurement as a decision by itself: a correction validated for one starved head,
applied globally, cost every head that was not starved. `structured_losses.py` said it
first - "measure the unweighted baseline before reaching for class weighting or focal loss"
- and this is that principle applied a second time, to a different architecture, rather than
carrying 27.62's fix over on the strength of a superficially similar imbalance number.

**The model is homr's own** - `CamVidModel` from `training/architecture/segmentation/model.py`,
a U-Net over a resnet18 encoder, already used for staff and notehead segmentation and reused
here rather than reinvented, per 27.69's reasoning. Its loss is multiclass Dice, which is not
cross-entropy: Dice is computed from per-class overlap (intersection over union-shaped), so a
class that covers a tiny fraction of every image does not automatically vanish into a
loss term dominated by background the way per-token cross-entropy does. Whether that is
enough on its own, for imbalance this severe, is the thing this file measures rather than
assumes - in either direction.

**Training uses `CamVidModel` as a plain `nn.Module`, not through PyTorch Lightning.**
`training/segmentation/train.py` drives it with `pl.Trainer.fit()`; every other training
script in this project - `train_structured_heads.py`, `train_recognizer.py` - is a plain
loop, and introducing Lightning for one corner of one track would be a second training
idiom for no measured benefit. `CamVidModel` is a `LightningModule`, which is still an
`nn.Module`; nothing about calling `.forward()` and `.parameters()` directly requires the
part of it this file does not use.
"""

# flake8: noqa: T201

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from training.architecture.segmentation.model import CamVidModel
from training.ocr.detector_masks import CLASS_ORDER
from training.ocr.detector_patches import (
    POSITIVE_RATIO,
    DetectorPatches,
    ImageBlockSampler,
    Sample,
    class_draw_weights,
    class_page_counts,
    read_index,
)

#: Background plus every detection class.
NUM_CLASSES = len(CLASS_ORDER) + 1
CLASS_NAMES = ("background", *CLASS_ORDER)


def collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    images = torch.stack(
        [torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 for image, _ in batch]
    )
    masks = torch.stack([torch.from_numpy(mask).long() for _, mask in batch])
    return {"images": images, "masks": masks}


def per_class_iou(
    model: CamVidModel, logits: torch.Tensor, masks: torch.Tensor
) -> dict[str, float]:
    """IoU for every class the model has, whether or not it appeared in this batch."""
    predicted = logits.softmax(dim=1).argmax(dim=1)
    tp, fp, fn, _ = smp.metrics.get_stats(
        predicted, masks, mode="multiclass", num_classes=NUM_CLASSES
    )
    iou = smp.metrics.iou_score(tp, fp, fn, torch.zeros_like(tp), reduction="none")
    # iou_score returns one row per image in the batch; average over the batch, per class,
    # and let 0/0 (a class absent from every image in this batch) read as "no data" rather
    # than a false zero - a class truly starved and a class merely unlucky this batch look
    # identical otherwise.
    present = tp.sum(dim=0) + fn.sum(dim=0) > 0
    mean_iou = iou.mean(dim=0)
    return {
        CLASS_NAMES[index]: float(mean_iou[index])
        for index in range(NUM_CLASSES)
        if present[index]
    }


def evaluate(model: CamVidModel, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    totals = collections.defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            masks = batch["masks"].to(device)
            logits = model(images)
            for name, value in per_class_iou(model, logits, masks).items():
                totals[name].append(value)
    return {name: sum(values) / len(values) for name, values in totals.items()}


def compute_class_weights(samples: list[Sample]) -> dict[str, float]:
    """`class_draw_weights` over every training mask's class presence - a one-time,
    read-every-mask-once pass at startup (not per epoch), the same information
    `class_page_counts`' own docstring says `box_centres_by_class` cannot answer since
    that function only ever sees one page at a time. Session note: built to test the
    still-untried half of 27.92's two-part diagnosis (see §1 item 1 in
    ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md) - "more distinct scores" (phase18) fixed
    Fingering but not the sampler-level competition; this is the other lever, not yet
    combined with phase18's data before this run.
    """
    masks = (cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE) for sample in samples)
    counts = class_page_counts(masks)
    print(f"class page counts: {counts}")
    weights = class_draw_weights(counts)
    print(f"class draw weights: { {k: round(v, 5) for k, v in weights.items()} }")
    return weights


def train(args: argparse.Namespace) -> dict:
    samples = read_index(args.index)
    class_weights = compute_class_weights(samples) if args.class_weighted_sampling else None
    dataset = DetectorPatches(
        samples,
        patches_per_image=args.patches_per_image,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
        class_weights=class_weights,
    )
    # A plain `shuffle=True` scatters one image's patches randomly across the whole epoch,
    # defeating DetectorPatches' one-slot decode cache - up to `patches_per_image` reads
    # of the same full-resolution page per epoch instead of one. ImageBlockSampler shuffles
    # image order but keeps one image's patches consecutive, so the cache actually hits.
    sampler = ImageBlockSampler(len(samples), args.patches_per_image, seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        collate_fn=collate,
    )
    valid_loader = None
    if args.valid_index:
        valid_samples = read_index(args.valid_index)
        # Fewer patches per image and no jitter randomness across epochs would be nicer,
        # but a fixed seed already makes every epoch's validation set the same crops -
        # good enough to compare epochs against each other, not meant as a held-out score.
        valid_dataset = DetectorPatches(
            valid_samples, patches_per_image=args.patches_per_image, seed=12345
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate,
        )

    model = CamVidModel(
        arch="Unet", encoder_name="resnet18", in_channels=3, out_classes=NUM_CLASSES,
        skip_weights_download=args.skip_pretrained,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"{len(dataset):,} patches from {len(samples):,} images, {NUM_CLASSES} classes")
    total_batches = -(-len(dataset) // args.batch_size)  # ceil div

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = collections.defaultdict(list)
        running_loss = 0.0
        batches = 0
        for batch in loader:
            if batches % 50 == 0:
                print(f"  epoch {epoch} batch {batches}/{total_batches}")
            images = batch["images"].to(args.device)
            masks = batch["masks"].to(args.device)
            logits = model(images)
            loss = model.loss_fn(logits.contiguous(), masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1
            with torch.no_grad():
                for name, value in per_class_iou(model, logits.detach(), masks).items():
                    totals[name].append(value)

        per_class = {name: sum(values) / len(values) for name, values in totals.items()}
        missing = [name for name in CLASS_NAMES if name not in per_class]
        print(f"epoch {epoch}: loss {running_loss / max(1, batches):.4f}")
        for name in CLASS_NAMES:
            if name in per_class:
                print(f"  {name:<14} IoU {per_class[name]:.3f}")
        if missing:
            print(f"  no data this epoch for: {', '.join(missing)}")
        record = {"epoch": epoch, "loss": running_loss / max(1, batches), **per_class}

        if valid_loader is not None:
            valid_per_class = evaluate(model, valid_loader, args.device)
            print("  valid:")
            for name in CLASS_NAMES:
                if name in valid_per_class:
                    print(f"    {name:<14} IoU {valid_per_class[name]:.3f}")
            record["valid"] = valid_per_class

        history.append(record)

    if args.weights:
        args.weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.weights)
    return {"history": history, "classes": list(CLASS_NAMES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="detector_masks index.txt")
    parser.add_argument("--valid-index", type=Path, help="detector_split.py's valid_index.txt")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument(
        "--positive-ratio", type=float, default=POSITIVE_RATIO,
        help=(
            "Share of drawn training patches centred on a box rather than a random "
            "page location (DetectorPatches.positive_ratio). 27.87's leading, still-"
            "untested hypothesis for Tempo/StaffText/Expression's precision collapse is "
            "that this needs to move toward true page frequency, or be made class-"
            "dependent - not exposed until now, since nothing had tried it."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class-weighted-sampling", action="store_true",
        help=(
            "Weight DetectorPatches' per-page positive-class choice by inverse distinct "
            "-page count (class_draw_weights), so a class spread across far more pages "
            "than another does not also win a proportionally larger share of positive "
            "draws corpus-wide. Off by default - preserves the existing uniform-among-"
            "present-classes behaviour."
        ),
    )
    parser.add_argument(
        "--skip-pretrained", action="store_true",
        help="Skip downloading imagenet encoder weights (offline runs, or tests).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report = train(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
