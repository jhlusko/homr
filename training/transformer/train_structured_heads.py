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
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from homr.transformer.capability_manifest import build as build_manifest
from homr.transformer.configs import Config
from training.architecture.transformer.checkpoint_loading import load_checkpoint
from training.architecture.transformer.structured_heads import head_names
from training.architecture.transformer.structured_losses import (
    StructuredLoss,
    structured_loss,
)

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


def structured_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for name, p in model.named_parameters() if name.startswith(NEW_PARAMETER_PREFIXES)]


def load_pinned(model: nn.Module, checkpoint: Path) -> None:
    """Load the pretrained weights, allowing only the new heads to be missing."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    report = load_checkpoint(model, state, NEW_PARAMETER_PREFIXES)
    print(f"checkpoint: {report.describe()}")


def train_epoch(
    model: nn.Module,
    batches: object,
    optimizer: torch.optim.Optimizer,
    head_targets: list[str],
    epoch: int,
) -> EpochReport:
    """One pass, updating only the structured heads."""
    model.train()
    totals: dict[str, float] = dict.fromkeys(head_targets, 0.0)
    support: dict[str, int] = dict.fromkeys(head_targets, 0)
    total = 0.0
    count = 0

    for batch in batches:  # type: ignore[attr-defined]
        targets = {name: batch[name] for name in head_targets if name in batch}
        if not targets:
            # Nothing to learn from this batch: a corpus without sidecars, or a crop with
            # no notes. Skipping keeps it out of the mean rather than averaging in a zero.
            continue
        outputs = model(**{k: v for k, v in batch.items() if k not in head_targets})
        logits = outputs["structured_logits"]
        if logits is None:
            raise ValueError("model has no structured heads - enable them in the config")

        result: StructuredLoss = structured_loss(logits, targets)
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
    config: Config,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="Dataset index.txt.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pinned .pth to start from.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the manifest.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    raise SystemExit(
        "Not runnable yet: this needs an OSSQ training set with notation sidecars, which "
        "needs the synthetic partwise staff crops that no pipeline run has produced. See "
        "27.11 - run omr-data-preprocessor's synthetic partwise cropping, then "
        "training/omr_datasets/convert_ossq.py, then remove this guard.\n"
        f"(index={args.index}, checkpoint={args.checkpoint}, out={args.out})"
    )


if __name__ == "__main__":
    main()
