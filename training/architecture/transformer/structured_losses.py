"""
Losses for the beam, stem and slur heads, and the masking that makes them mean anything.

Almost every position in a decoded sequence has nothing to say about beaming. Barlines,
clefs and key signatures are not notes; a quarter note has no beam levels; a sixteenth
has two and not six; a source that omits <stem> has no stem answer to score against. Left
unmasked, every one of those becomes a free correct prediction of NOT_APPLICABLE, and the
head reports high accuracy for learning the shape of the vocabulary rather than reading
the page.

So each head is scored only where its target is real:

  beam level N   only on notes whose duration has at least N flags
  stem           only on notes whose source states a direction
  slur event     on every note
  slur side      only where the source states a placement, which is about half of spans

Support is counted per class alongside the loss, because these classes are wildly
unbalanced - 608,166 level-1 beams against 1,697 level-4 in training - and a mean loss
says nothing about whether the rare ones moved. The design's instruction is to measure
the unweighted baseline before reaching for class weighting or focal loss, so this
computes plain cross-entropy and reports what would justify the alternative.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F  # noqa: N812

#: Target value meaning "no supervision here". Matches the decoder's own convention so
#: the two masking schemes cannot drift apart.
IGNORE_INDEX = -100


@dataclass
class HeadLoss:
    name: str
    loss: torch.Tensor
    #: Positions that carried a real target. A head can be genuinely absent from a batch -
    #: level-4 beams appear in well under 1% of notes - and a zero loss over zero
    #: positions is not the same as a zero loss over many.
    supervised: int
    support: dict[int, int] = field(default_factory=dict)


@dataclass
class StructuredLoss:
    total: torch.Tensor
    heads: list[HeadLoss]

    def supervised_heads(self) -> list[HeadLoss]:
        return [head for head in self.heads if head.supervised > 0]

    def describe(self) -> str:
        parts = [
            f"{head.name}={head.loss.item():.4f}({head.supervised})"
            for head in self.supervised_heads()
        ]
        unsupervised = [head.name for head in self.heads if head.supervised == 0]
        text = "  ".join(parts) or "no supervised heads in this batch"
        if unsupervised:
            text += f"  [no targets: {', '.join(unsupervised)}]"
        return text


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Cross-entropy over positions whose target is not IGNORE_INDEX.

    Returns a zero loss when nothing is supervised, rather than a NaN mean over an empty
    set, so a batch without a rare class does not poison the total.
    """
    supervised = int((targets != IGNORE_INDEX).sum().item())
    if supervised == 0:
        return logits.sum() * 0.0, 0
    return (
        F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="mean",
        ),
        supervised,
    )


def class_support(targets: torch.Tensor, num_classes: int) -> dict[int, int]:
    """How many supervised positions fall in each class."""
    valid = targets[targets != IGNORE_INDEX]
    if valid.numel() == 0:
        return {}
    counts = torch.bincount(valid.reshape(-1), minlength=num_classes)
    return {index: int(count) for index, count in enumerate(counts.tolist()) if count}


def structured_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> StructuredLoss:
    """Sum the per-head losses over the heads the model actually has.

    Heads without targets in this batch contribute nothing and are reported separately,
    rather than being averaged in as zeros - which would make the total drift with how
    often rare classes happen to appear rather than with how well they are predicted.

    A target for a head the model does not have is an error, not something to skip: it
    means the label pipeline and the model disagree about the configuration, and silently
    dropping the supervision would leave that head untrained with nothing to show for it.
    """
    unknown = sorted(set(targets) - set(logits))
    if unknown:
        raise KeyError(
            f"targets for head(s) the model does not have: {unknown} - "
            "label configuration and model configuration disagree"
        )

    scale = weights or {}
    heads: list[HeadLoss] = []
    total = None
    for name, head_logits in logits.items():
        if name not in targets:
            continue
        loss, supervised = masked_cross_entropy(head_logits, targets[name])
        heads.append(
            HeadLoss(
                name=name,
                loss=loss,
                supervised=supervised,
                support=class_support(targets[name], head_logits.shape[-1]),
            )
        )
        if supervised:
            weighted = loss * scale.get(name, 1.0)
            total = weighted if total is None else total + weighted

    if total is None:
        anything = next(iter(logits.values()))
        total = anything.sum() * 0.0
    return StructuredLoss(total=total, heads=heads)
