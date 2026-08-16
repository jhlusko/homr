"""
Turn head logits and target tensors back into notation, so the two can be compared.

The metrics in `training/transformer/structured_metrics.py` are written against
`NoteNotation` sequences rather than tensors, because two of them - exact beam-vector
match and slur endpoint pairing - are questions about a note or a span, not about a
position. This is the bridge.

The reference side is rebuilt from the *target* tensors rather than read back from the
sidecar. That is deliberate: the targets are what training actually saw, masking and all,
so a scoring pass built from them cannot flatter the model by grading it on positions the
loss never touched, or disagree with training about which positions those were.

Predictions are masked to the same positions. An argmax exists at every position,
including barlines, clefs and padding, but a "slur" predicted on a barline is not a
prediction the model was ever asked to make - counting it would measure the mask rather
than the head.
"""

from collections.abc import Mapping, Sequence
from typing import TypeVar

import torch

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    MAX_BEAM_LEVELS,
    MAX_SLUR_SLOTS,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
)
from training.architecture.transformer.structured_heads import (
    BEAM_HEAD,
    SLUR_EVENT_HEAD,
    SLUR_SIDE_HEAD,
    STEM_HEAD,
)
from training.architecture.transformer.structured_losses import IGNORE_INDEX

T = TypeVar("T")


def decode_reference(
    targets: Mapping[str, torch.Tensor], beam_levels: int, slur_slots: int
) -> list[list[NoteNotation]]:
    """The notation the targets encode, one sequence per batch row.

    Masked positions come back as the state that means "no answer here" - the same state
    the metrics skip - so a masked position cannot be scored as a correct prediction.
    """
    return _decode(dict(targets), _shape(targets), beam_levels, slur_slots)


def decode_predictions(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    beam_levels: int,
    slur_slots: int,
) -> list[list[NoteNotation]]:
    """The notation the heads predict, masked to the positions the targets supervise."""
    predicted = {name: tensor.argmax(dim=-1) for name, tensor in logits.items()}
    masked = {
        name: torch.where(targets[name] == IGNORE_INDEX, IGNORE_INDEX, tensor)
        for name, tensor in predicted.items()
        if name in targets
    }
    # The shape comes from the targets, never from the logits: a run that trained only
    # some heads has logits for only those, and a run that trained none still has to
    # produce a sequence of "nothing predicted" rather than fail to decode.
    return _decode(masked, _shape(targets), beam_levels, slur_slots)


def _shape(tensors: Mapping[str, torch.Tensor]) -> tuple[int, int]:
    if not tensors:
        raise ValueError("no target tensors, so there is no sequence to decode against")
    any_head = next(iter(tensors.values()))
    return int(any_head.shape[0]), int(any_head.shape[1])


def _decode(
    indices: dict[str, torch.Tensor],
    shape: tuple[int, int],
    beam_levels: int,
    slur_slots: int,
) -> list[list[NoteNotation]]:
    rows, length = shape
    return [
        [_note(indices, row, column, beam_levels, slur_slots) for column in range(length)]
        for row in range(rows)
    ]


def _note(
    indices: dict[str, torch.Tensor],
    row: int,
    column: int,
    beam_levels: int,
    slur_slots: int,
) -> NoteNotation:
    beams = [
        _lookup(
            indices,
            BEAM_HEAD.format(level=level),
            row,
            column,
            BEAM_LEVEL_CLASSES,
            BeamLevelState.NOT_APPLICABLE,
        )
        for level in range(1, beam_levels + 1)
    ]
    beams.extend([BeamLevelState.NOT_APPLICABLE] * (MAX_BEAM_LEVELS - len(beams)))

    slurs = [
        (
            _lookup(
                indices,
                SLUR_EVENT_HEAD.format(slot=slot),
                row,
                column,
                SLUR_EVENT_CLASSES,
                SlurEvent.NONE,
            ),
            _lookup(
                indices,
                SLUR_SIDE_HEAD.format(slot=slot),
                row,
                column,
                SLUR_SIDE_CLASSES,
                SlurSide.UNSPECIFIED,
            ),
        )
        for slot in range(1, slur_slots + 1)
    ]
    slurs.extend([(SlurEvent.NONE, SlurSide.UNSPECIFIED)] * (MAX_SLUR_SLOTS - len(slurs)))

    return NoteNotation(
        beam_levels=tuple(beams),
        stem=_lookup(indices, STEM_HEAD, row, column, STEM_CLASSES, StemDirection.UNKNOWN),
        slurs=tuple(slurs),
    )


def _lookup(
    indices: dict[str, torch.Tensor],
    name: str,
    row: int,
    column: int,
    classes: Sequence[T],
    absent: T,
) -> T:
    """One head's class at one position, or `absent` where there is nothing to score.

    `absent` is the state the metrics treat as "not asked" - NOT_APPLICABLE for a beam
    level, UNKNOWN for a stem - so an untrained head and a masked position both drop out
    of the figures instead of being scored as answers.
    """
    if name not in indices:
        return absent
    value = int(indices[name][row, column])
    if value == IGNORE_INDEX:
        return absent
    return classes[value]
