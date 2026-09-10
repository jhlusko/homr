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


def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int]:
    """Cross-entropy with the two corrections for a class the model can afford to ignore.

    `gamma` is the focal exponent: each position is scaled by (1 - p_true)^gamma, so a
    `none` the model already predicts with confidence 0.999 contributes almost nothing and
    stops drowning out the rare classes. `alpha` is a per-class weight vector, the blunter
    instrument. Both at their defaults - gamma 0, alpha None - this *is* `cross_entropy`,
    which a test pins, so the unweighted baseline stays reachable and comparable.

    Why both: upstream issue #61 records dynamics at ~0.05% of tokens collapsing to
    never-predicted, taking SER from 26% to 132%, and the discussion there converged on
    focal loss - as oemer's UNet uses - or class weights at 50x, without settling which.
    27.49 found our tie head in the same condition at 0.109%, so this is the instrument for
    a sweep rather than a single guess.
    """
    supervised = int((targets != IGNORE_INDEX).sum().item())
    if supervised == 0:
        return logits.sum() * 0.0, 0

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    keep = flat_targets != IGNORE_INDEX
    flat_logits, flat_targets = flat_logits[keep], flat_targets[keep]

    log_probabilities = F.log_softmax(flat_logits, dim=-1)
    picked = log_probabilities.gather(1, flat_targets.unsqueeze(1)).squeeze(1)

    losses = -picked
    if gamma:
        losses = losses * (1.0 - picked.exp()).pow(gamma)
    if alpha is not None:
        losses = losses * alpha.to(losses.device)[flat_targets]
        # Normalised by the weight actually applied, not by the count, so raising a rare
        # class's weight does not also raise the head's share of the total loss - the two
        # knobs stay independent.
        return losses.sum() / alpha.to(losses.device)[flat_targets].sum(), supervised
    return losses.mean(), supervised


def inverse_frequency_alpha(
    support: dict[int, int], num_classes: int, cap: float = 50.0
) -> torch.Tensor:
    """Per-class weights from observed counts, normalised to mean 1 and capped.

    Uncapped inverse frequency on the tie head would weight `start` about 900x against
    `none`, which trades one collapse for another - the head would predict ties everywhere.
    50 is the cap because it is the figure the upstream attempt used, so a result here is
    comparable to that one rather than to nothing.

    **The cap flattens the rarest classes together**, and that is a real consequence rather
    than an accident. On the tie head `start` at 0.109% and `start_and_stop` at 0.014% both
    exceed it, so both come out at the same weight even though one is eight times rarer.
    Every class past the cap is simply "as boosted as this scheme goes"; if the ordering
    among them turns out to matter, the fix is a gentler curve such as inverse square root,
    not a higher cap, which is the collapse this cap exists to prevent.
    """
    counts = torch.tensor(
        [max(1, support.get(index, 0)) for index in range(num_classes)], dtype=torch.float
    )
    weights = (counts.sum() / counts).clamp(max=cap)
    return weights / weights.mean()


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
    gamma: float | dict[str, float] = 0.0,
    alpha: dict[str, torch.Tensor] | None = None,
) -> StructuredLoss:
    """Sum the per-head losses over the heads the model actually has.

    Heads without targets in this batch contribute nothing and are reported separately,
    rather than being averaged in as zeros - which would make the total drift with how
    often rare classes happen to appear rather than with how well they are predicted.

    A target for a head the model does not have is an error, not something to skip: it
    means the label pipeline and the model disagree about the configuration, and silently
    dropping the supervision would leave that head untrained with nothing to show for it.

    `gamma` and `alpha` reach the per-position loss; `weights` scales whole heads against
    each other. They are separate knobs on purpose - the first two are about a class the
    model can afford to ignore inside one head, the third is about how much one head
    matters against another - and 27.49 is about the first problem only.

    `gamma` as a single number applies to every head, which phase12 measured the cost of:
    beam and slur lost 1 to 8 points across all three domains, including synthetic and
    PDMX where nothing else about training changed, while ties gained 0.6 to 2.7. Only the
    tie head was starved; every head paid focal's discount on its own confident positions.
    A dict scopes gamma per head name so the correction can be applied only where it was
    diagnosed, defaulting absent heads to 0.0 - ordinary cross-entropy, not a guess at what
    they might need.
    """
    unknown = sorted(set(targets) - set(logits))
    if unknown:
        raise KeyError(
            f"targets for head(s) the model does not have: {unknown} - "
            "label configuration and model configuration disagree"
        )

    scale = weights or {}
    per_head_gamma = gamma if isinstance(gamma, dict) else None
    heads: list[HeadLoss] = []
    total = None
    for name, head_logits in logits.items():
        if name not in targets:
            continue
        head_gamma = per_head_gamma.get(name, 0.0) if per_head_gamma is not None else gamma
        loss, supervised = focal_cross_entropy(
            head_logits, targets[name], gamma=head_gamma, alpha=(alpha or {}).get(name)
        )
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
