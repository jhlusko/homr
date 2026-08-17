"""
Phase 2: train the beam, stem and slur heads over a frozen core.

The question this run answers is narrow and worth keeping narrow: does the pretrained
representation already carry enough visual evidence to learn explicit beaming, stem
direction and richer slurs? Only the new projections move. If anything else did, a gain
could be the heads or the core drifting to suit them, and the answer would not be an
answer.

Three things follow from that and are enforced rather than assumed. The checkpoint loads
under an allowlist, so an absent core weight is an error rather than a silent
reinitialisation. The existing objective is not touched - the structured loss is its own
number and never joins `loss`. And the manifest written at the end declares only the
heads this run actually optimised, so a projection that never saw a gradient cannot be
mistaken for a capability.

Per-head support is logged every epoch alongside the loss. These classes are extremely
unbalanced - hundreds of thousands of level-1 beams against a few hundred level-4 - and a
falling mean loss says nothing about whether the rare ones moved, or whether a head had
any targets at all.
"""

# flake8: noqa: T201

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader

from homr.transformer.capability_manifest import build as build_manifest
from training.architecture.transformer.checkpoint_loading import load_checkpoint
from training.architecture.transformer.structured_heads import head_names
from training.architecture.transformer.structured_losses import (
    IGNORE_INDEX,
    StructuredLoss,
    structured_loss,
)
from training.architecture.transformer.structured_targets import align_to_decoder_output

# The model, the image pipeline and the dataset loader are imported inside the functions
# that need them rather than here. Everything above this line is pure - the epoch loop,
# the collate, the manifest - and importing this module for those pulls in timm,
# albumentations and transformers through TrOMR if the imports sit at the top. The tests
# for the pure parts should not need the whole training stack installed to run.

#: Parameters the pretrained checkpoint cannot be expected to contain.
NEW_PARAMETER_PREFIXES = ("decoder.structured_heads.",)


@dataclass
class EpochReport:
    epoch: int
    total: float
    batches: int
    per_head: dict[str, float]
    support: dict[str, int]

    def describe(self) -> str:
        silent = [name for name, count in self.support.items() if count == 0]
        parts = " ".join(
            f"{name}={self.per_head[name]:.4f}({self.support[name]})"
            for name in sorted(self.per_head)
            if self.support[name] > 0
        )
        line = f"epoch {self.epoch}: mean {self.total / max(self.batches, 1):.4f}  {parts}"
        if silent:
            line += f"  [no targets all epoch: {', '.join(sorted(silent))}]"
        return line


def set_probe_mode(model: nn.Module) -> None:
    """Frozen core in eval mode, heads in train mode.

    `model.train()` would leave the core's dropout on - homr's decoder carries 0.1 on
    attention, feed-forward and whole layers - so the heads would learn from a
    stochastically perturbed hidden state and then meet an unperturbed one at inference.
    The point of this experiment is to ask what the pretrained representation already
    carries, and that representation is the one the model produces in eval mode.

    Dropout in a frozen backbone is defensible as augmentation when the backbone is being
    tuned too. Here nothing about the core is being learned, so the noise is not
    regularising anything - it is only putting a gap between what the heads train on and
    what they are scored on.
    """
    model.eval()
    heads = getattr(getattr(model, "decoder", None), "structured_heads", None)
    if heads is not None:
        heads.train()


def structured_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.startswith(NEW_PARAMETER_PREFIXES)]


def load_pinned(model: nn.Module, checkpoint: Path) -> None:
    """Load the pretrained weights, allowing only the new heads to be missing."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = load_checkpoint(model, state, NEW_PARAMETER_PREFIXES)
    print(f"checkpoint: {report.describe()}")


def measure_class_weights(
    batches: object,
    head_targets: Sequence[str],
    model: nn.Module,
    device: str = "cpu",
    cap: float = 50.0,
) -> dict[str, torch.Tensor]:
    """Count each head's classes over the training data, then invert.

    A separate pass rather than per-batch counts. A batch that happens to hold no `start`
    tie would give that class a weight from a sample of zero, and the weights would then
    wobble from step to step - the model would be chasing a moving objective on exactly the
    classes it is least able to learn.
    """
    from training.architecture.transformer.structured_losses import (
        class_support,
        inverse_frequency_alpha,
    )

    # The width of the logits is the authority on how many classes a head has. Counting
    # the classes that appear in the targets instead would size each head by whatever the
    # corpus happened to contain, and a class absent from training - `start_and_stop`
    # appears 293 times in two million - could drop out of the weight vector entirely.
    # Taking it from the logits rather than the module names avoids restating the mapping
    # between the two, which is exactly the kind of duplicated correspondence that has
    # gone wrong three times in this work.
    sizes: dict[str, int] = {}
    counts: dict[str, dict[int, int]] = {name: {} for name in head_targets}
    for raw in batches:  # type: ignore[attr-defined]
        targets = {name: raw[name] for name in head_targets if name in raw}
        if not targets:
            continue
        if not sizes:
            # Moved to the model's device first. The training loop does this and this pass
            # did not, which is a mismatch no unit test would see - both run on cpu there.
            moved = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in raw.items()
                if key not in head_targets
            }
            with torch.no_grad():
                probe = model(**moved)
            sizes = {
                name: tensor.shape[-1]
                for name, tensor in (probe["structured_logits"] or {}).items()
            }
        targets = align_to_decoder_output(targets)
        for name, tensor in targets.items():
            if name in sizes:
                for index, count in class_support(tensor, sizes[name]).items():
                    counts[name][index] = counts[name].get(index, 0) + count

    return {
        name: inverse_frequency_alpha(support, sizes[name], cap=cap)
        for name, support in counts.items()
        if support and name in sizes
    }


def train_epoch(
    model: nn.Module,
    batches: object,
    optimizer: torch.optim.Optimizer,
    head_targets: Sequence[str],
    epoch: int,
    device: str = "cpu",
    gamma: float = 0.0,
    alpha: dict[str, torch.Tensor] | None = None,
) -> EpochReport:
    """One pass, updating only the structured heads.

    `gamma` and `alpha` are the class-imbalance controls of 27.49. Both default to off, so
    the unweighted baseline every earlier result was measured against stays the default and
    a sweep changes one thing at a time.
    """
    set_probe_mode(model)
    totals: dict[str, float] = dict.fromkeys(head_targets, 0.0)
    support: dict[str, int] = dict.fromkeys(head_targets, 0)
    total = 0.0
    count = 0

    for raw in batches:  # type: ignore[attr-defined]
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw.items()}
        targets = {name: batch[name] for name in head_targets if name in batch}
        if not targets:
            # Nothing to learn from this batch: a corpus without sidecars, or a crop with
            # no notes. Skipping keeps it out of the mean rather than averaging in a zero.
            continue
        # The heads read the decoder's hidden state, which predicts the *next* token, so
        # the labels have to move with it. Without this every head trains on the token
        # after the one it describes.
        targets = align_to_decoder_output(targets)
        if all(int((t != IGNORE_INDEX).sum()) == 0 for t in targets.values()):
            continue
        outputs = model(**{k: v for k, v in batch.items() if k not in head_targets})
        logits = outputs["structured_logits"]
        if logits is None:
            raise ValueError("model has no structured heads - enable them in the config")

        result: StructuredLoss = structured_loss(logits, targets, gamma=gamma, alpha=alpha)
        optimizer.zero_grad(set_to_none=True)
        result.total.backward()
        optimizer.step()

        total += float(result.total.item())
        count += 1
        for head in result.heads:
            totals[head.name] += float(head.loss.item())
            support[head.name] += head.supervised

    per_head = {name: totals[name] / max(count, 1) for name in totals}
    return EpochReport(epoch, total, count, per_head, support)


def write_manifest(
    path: Path,
    # Duck-typed like capability_manifest.build: only the limits and head counts are
    # read, so a test can pass a stand-in without constructing the whole Config.
    config: Any,
    trained: tuple[str, ...],
    model_revision: str,
    training_revision: str,
    run_id: str,
) -> None:
    manifest = build_manifest(
        config=config,
        trained_heads=trained,
        available_heads=tuple(
            head_names(config.structured_beam_levels, config.structured_slur_slots)
        ),
        model_revision=model_revision,
        training_revision=training_revision,
        label_schema_version="homr.structured-symbols.v1",
        run_id=run_id,
    )
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"manifest: {len(manifest.supported_heads)} head(s) declared -> {path}")


def heads_with_support(reports: list[EpochReport]) -> tuple[str, ...]:
    """Heads that saw at least one target across the run.

    A head that never had a target was not trained, whatever the architecture provided,
    and declaring it would advertise a projection still holding its initialisation.
    """
    seen: dict[str, int] = {}
    for report in reports:
        for name, count in report.support.items():
            seen[name] = seen.get(name, 0) + count
    return tuple(name for name, count in sorted(seen.items()) if count > 0)


def collate(items: list[dict[str, Any]], head_targets: Sequence[str]) -> dict[str, Any]:
    """Batch items that may or may not carry notation targets.

    A corpus can mix annotated and unannotated examples - the wrapper adds no keys at all
    where there is no sidecar - and the default collate would reject a batch whose items
    have different keys. Dropping the structured keys for such a batch would be worse than
    the error: the annotated examples in it would silently stop being supervised. So a
    missing example is filled with IGNORE_INDEX and contributes nothing, while its
    neighbours are still learned from.
    """
    batch: dict[str, Any] = {}
    shared = [key for key in items[0] if key not in head_targets]
    for key in shared:
        batch[key] = torch.stack([item[key] for item in items])

    for name in head_targets:
        present = [item[name] for item in items if name in item]
        if not present:
            continue
        blank = torch.full_like(present[0], IGNORE_INDEX)
        batch[name] = torch.stack([item.get(name, blank) for item in items])
    return batch


def _target_names(config: Any) -> list[str]:
    from training.transformer.structured_dataset import target_names

    return target_names(config.structured_beam_levels, config.structured_slur_slots)


def build_batches(
    index: Path,
    config: Any,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
    validation: bool = False,
) -> tuple[TorchDataLoader, int]:
    """The loader, wrapped so each item carries its notation targets.

    `shuffle` is off for evaluation, where batches have to arrive in index order: an
    analysis that names which example a prediction came from can only do so by counting
    along, and a shuffled loader would silently attach every prediction to the wrong
    staff. `validation` additionally turns off image augmentation.
    """
    from training.transformer.data_loader import load_dataset
    from training.transformer.structured_dataset import StructuredNotationDataset

    samples = index.read_text(encoding="utf-8").splitlines()
    # val_split routes the whole index to the loader built with is_validation=True, which
    # turns off the distortion and occlusion the training loader applies. Scoring a held
    # out set through the training loader would measure the model on images that are not
    # the ones it will meet, and would not even be reproducible run to run.
    datasets = load_dataset(
        [line for line in samples if line.strip()],
        config,
        val_split=1.0 if validation else 0.0,
    )
    wrapped = StructuredNotationDataset(
        datasets["validation" if validation else "train"],
        config.structured_beam_levels,
        config.structured_slur_slots,
    )
    names = _target_names(config)
    loader = TorchDataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=lambda items: collate(items, names),
    )
    return loader, len(wrapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned .pth to start from.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the manifest.")
    parser.add_argument("--weights", type=Path, help="Where to write the trained head weights.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help="Focal exponent. 0 is plain cross-entropy, the baseline every earlier "
        "result used. 2 is the usual starting point (27.49).",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Weight each head's classes by inverse frequency, measured from a first "
        "pass over the training data (27.49).",
    )
    parser.add_argument(
        "--weight-cap",
        type=float,
        default=50.0,
        help="How far --class-weights may rebalance. On the tie head this moves the tie "
        "classes from 0.23%% of the gradient unweighted to 2.3%% at 10, 10.4%% at 50 and "
        "31.6%% at 200; uncapped is 75%%, where 293 examples carry a quarter of it.",
    )
    args = parser.parse_args()

    from homr.transformer.configs import Config
    from training.architecture.transformer.tromr_arch import TrOMR
    from training.run_id import get_run_id

    config = Config()
    config.enable_structured_heads = True
    names = _target_names(config)

    model = TrOMR(config)
    load_pinned(model, args.checkpoint)
    trainable = model.freeze_core_for_structured_heads()
    print(f"training {len(trainable)} tensor(s), everything else frozen")
    model.to(args.device)

    batches, examples = build_batches(args.index, config, args.batch_size, args.workers)
    print(f"{examples} example(s) from {args.index}")

    alpha = None
    if args.class_weights:
        alpha = measure_class_weights(
            batches, names, model, device=args.device, cap=args.weight_cap
        )
        for name, vector in sorted(alpha.items()):
            spread = vector.max().item() / vector.min().item()
            print(f"  {name}: class weights spread {spread:.1f}x")

    optimizer = torch.optim.Adam(structured_parameters(model), lr=args.lr)
    reports = []
    for epoch in range(1, args.epochs + 1):
        report = train_epoch(
            model, batches, optimizer, names, epoch,
            device=args.device, gamma=args.focal_gamma, alpha=alpha,
        )
        print(report.describe())
        reports.append(report)

    trained = heads_with_support(reports)
    if args.weights:
        torch.save(model.decoder.structured_heads.state_dict(), args.weights)
        print(f"weights: {args.weights}")
    write_manifest(
        args.out,
        config,
        trained,
        model_revision=str(args.checkpoint),
        training_revision=git_revision(),
        run_id=args.run_id or get_run_id(),
    )


def git_revision() -> str:
    """The training-side revision recorded in the manifest.

    Best effort: a manifest from a tree that is not a git checkout is still worth having,
    so an unavailable revision is recorded as unknown rather than failing the run.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


if __name__ == "__main__":
    main()
