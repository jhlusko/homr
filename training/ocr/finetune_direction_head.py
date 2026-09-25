"""Pilot DirectionText head-only adaptation from the released seven-class e4 detector.

The released detector has good real OSSQ Dynamic recall but no direction recall. Keep
its encoder, decoder, and non-direction output filters fixed, remap its output to the
five-class scheme, and train only the DirectionText filter on the existing disjoint
real/synthetic patch banks. This is an experiment, not a release decision: select and
test its epochs with the frozen real-page gate in docs/DETECTOR_REAL_PAGE_RELEASE_GATE_2026-09-25.md.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from homr.text_detector_classes import (
    DIRECTION_CLASS_ORDER,
    LEGACY_CLASS_ORDER,
    read_class_order,
    write_class_order,
)
from training.architecture.segmentation.model import CamVidModel
from training.ocr.detector_patches import PreExtractedPatches, read_index
from training.ocr.train_detector import collate, evaluate

E4_SHA256 = "fcb8ca529fa355b0427112bda1c4edd95b2d6cc6c369be0c32f3313987a4ec82"
HEAD_WEIGHT = "model.segmentation_head.0.weight"
HEAD_BIAS = "model.segmentation_head.0.bias"
SEED = 20260925


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remap_state(old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy e4's non-direction filters exactly; average its three text filters."""
    expected = ("background", *LEGACY_CLASS_ORDER)
    target = ("background", *DIRECTION_CLASS_ORDER)
    result = {key: value.clone() for key, value in old.items()}
    for key in (HEAD_WEIGHT, HEAD_BIAS):
        source = old[key]
        if source.shape[0] != len(expected):
            raise ValueError(f"{key}: expected {len(expected)} channels, got {source.shape[0]}")
        output = source.new_empty((len(target), *source.shape[1:]))
        for channel, label in enumerate(target):
            if label == "DirectionText":
                output[channel] = torch.stack(
                    [source[expected.index(name)] for name in ("Expression", "Tempo", "StaffText")]
                ).mean(dim=0)
            else:
                output[channel] = source[expected.index(label)]
        result[key] = output
    return result


def check_score_exclusion(
    page_audit: Path, split_manifest: Path, real_index: Path, synthetic_index: Path
) -> dict:
    audit = json.loads(page_audit.read_text())
    manifest = json.loads(split_manifest.read_text())
    held = {
        score
        for corpus in manifest["corpora"].values()
        for role in corpus["roles"].values()
        for score in role["scores"]
    }
    train = set(audit["train_scores"])
    if held & train:
        raise ValueError(f"training scores overlap frozen holdout: {sorted(held & train)}")
    if digest(Path(audit["initial_checkpoint"])) != audit["initial_checkpoint_sha256"]:
        raise ValueError("09-19 audit initial checkpoint digest changed")
    # The existing bank files came from that audit. Record their exact bytes, and
    # refuse unexpected paths rather than silently fitting on a newly mixed bank.
    real = read_index(real_index)
    synthetic = read_index(synthetic_index)
    if len(real) != audit["real_pages"] * 64 or len(synthetic) != audit["synthetic_patches"]:
        raise ValueError("patch-bank size differs from the audited 09-19 recipe")
    if any("/real-bank/" not in sample.image for sample in real):
        raise ValueError("unexpected real patch-bank path")
    return {
        "real_patches": len(real),
        "synthetic_patches": len(synthetic),
        "real_index_sha256": digest(real_index),
        "synthetic_index_sha256": digest(synthetic_index),
        "page_audit_sha256": digest(page_audit),
        "split_manifest_sha256": digest(split_manifest),
        "heldout_scores": len(held),
    }


def train(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if digest(args.e4_weights) != E4_SHA256:
        raise ValueError("e4 weight digest mismatch")
    if read_class_order(args.e4_weights) != LEGACY_CLASS_ORDER:
        raise ValueError("e4 class order mismatch")
    audit = check_score_exclusion(
        args.page_audit, args.split_manifest, args.real_index, args.synthetic_index
    )
    real = read_index(args.real_index)
    synthetic = read_index(args.synthetic_index)
    dataset = PreExtractedPatches(real + synthetic)
    sampler = WeightedRandomSampler(
        [0.5 / len(real)] * len(real) + [0.5 / len(synthetic)] * len(synthetic),
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, collate_fn=collate,
    )
    valid = DataLoader(
        PreExtractedPatches(read_index(args.valid_index)),
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate,
    )
    model = CamVidModel(
        arch="Unet", encoder_name="resnet18", in_channels=3,
        out_classes=len(DIRECTION_CLASS_ORDER) + 1, skip_weights_download=True,
    ).to(args.device)
    old = torch.load(args.e4_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(remap_state(old), strict=True)
    # Only channel 3 (DirectionText) is trainable. AdamW's weight decay is disabled
    # because it would alter fixed channels even after their gradients were zeroed.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.model.segmentation_head[0]
    direction_channel = DIRECTION_CLASS_ORDER.index("DirectionText") + 1
    fixed_weight = head.weight.detach().clone()
    fixed_bias = head.bias.detach().clone()
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    channel_mask = torch.zeros_like(head.bias)
    channel_mask[direction_channel] = 1
    head.weight.register_hook(lambda grad: grad * channel_mask[:, None, None, None])
    head.bias.register_hook(lambda grad: grad * channel_mask)
    optimizer = torch.optim.AdamW((head.weight, head.bias), lr=args.lr, weight_decay=0)
    loss_fn = smp.losses.DiceLoss("multiclass", from_logits=True, ignore_index=255)
    history = []
    metadata = {
        "recipe": "e4 frozen trunk and non-direction outputs; DirectionText output only",
        "e4_sha256": E4_SHA256,
        "class_order": DIRECTION_CLASS_ORDER,
        "audit": audit,
        "seed": SEED,
        "learning_rate": args.lr,
        "samples_per_epoch": args.samples_per_epoch,
    }
    (args.out / "recipe.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for epoch in range(1, args.epochs + 1):
        # Keep BatchNorm running statistics frozen with the e4 trunk. Gradients still
        # reach the trainable output filter while the module is in evaluation mode.
        model.eval()
        losses = []
        for batch_index, batch in enumerate(loader):
            logits = model(batch["images"].to(args.device))
            loss = loss_fn(logits, batch["masks"].to(args.device))
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                head.weight[:direction_channel].copy_(fixed_weight[:direction_channel])
                head.weight[direction_channel + 1:].copy_(fixed_weight[direction_channel + 1:])
                head.bias[:direction_channel].copy_(fixed_bias[:direction_channel])
                head.bias[direction_channel + 1:].copy_(fixed_bias[direction_channel + 1:])
            losses.append(float(loss))
            if batch_index % 50 == 0:
                print(f"epoch {epoch} batch {batch_index}/{len(loader)} loss {losses[-1]:.4f}", flush=True)
        checkpoint = args.out / f"epoch-{epoch}.pth"
        torch.save(model.state_dict(), checkpoint)
        write_class_order(checkpoint, DIRECTION_CLASS_ORDER)
        valid_scores = evaluate(model, valid, args.device, ignore_index=255)
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "valid": valid_scores})
        (args.out / "history.json").write_text(
            json.dumps({"classes": ["background", *DIRECTION_CLASS_ORDER], "history": history}, indent=2) + "\n"
        )
        print(json.dumps(history[-1]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--e4-weights", type=Path, required=True)
    parser.add_argument("--page-audit", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--real-index", type=Path, required=True)
    parser.add_argument("--synthetic-index", type=Path, required=True)
    parser.add_argument("--valid-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
