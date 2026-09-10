"""
Turn extracted notation into per-token targets, masked where there is nothing to learn.

The masking is the substance here, not the tensor shuffling. Three rules, each of which
would otherwise hand a head free correct answers:

A beam level only applies to a note whose duration has at least that many flags. A
quarter note has none, a sixteenth has two. Scoring level 3 on a sixteenth teaches
NOT_APPLICABLE, which the model can infer from the rhythm token it already predicts, and
inflates accuracy with positions that were never in question.

A stem direction only exists where the source states one. UNKNOWN marks a silent source;
it is masked rather than scored, so silence cannot be learned as an answer.

A slur side only exists where the source states a placement, which across this corpus is
about half of all spans. The event is always real - NONE is a genuine prediction - so
only the side is masked.

Positions that are not notes at all carry no notation target of any kind.
"""

from collections.abc import Sequence

import torch

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    DYNAMIC_CLASSES,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    TIE_CLASSES,
    BeamLevelState,
    NoteNotation,
    SlurSide,
    StemDirection,
    trained_dynamic_mark,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.structured_heads import (
    BEAM_HEAD,
    DYNAMIC_HEAD,
    SLUR_EVENT_HEAD,
    SLUR_SIDE_HEAD,
    STEM_HEAD,
    TIE_HEAD,
)
from training.architecture.transformer.structured_losses import IGNORE_INDEX

_BEAM_INDEX = {state: index for index, state in enumerate(BEAM_LEVEL_CLASSES)}
_STEM_INDEX = {state: index for index, state in enumerate(STEM_CLASSES)}
_EVENT_INDEX = {state: index for index, state in enumerate(SLUR_EVENT_CLASSES)}
_SIDE_INDEX = {state: index for index, state in enumerate(SLUR_SIDE_CLASSES)}
_TIE_INDEX = {state: index for index, state in enumerate(TIE_CLASSES)}
_DYNAMIC_INDEX = {state: index for index, state in enumerate(DYNAMIC_CLASSES)}


def build_targets(
    sequences: Sequence[Sequence[NoteNotation | None]],
    beam_levels: int,
    slur_slots: int,
) -> dict[str, torch.Tensor]:
    """Per-head target tensors of shape (batch, sequence).

    `sequences[b][t]` is the notation for token t of example b, or None where that token
    is not a note - a barline, a clef, padding - and therefore carries no notation target.
    """
    batch = len(sequences)
    length = max((len(sequence) for sequence in sequences), default=0)
    targets = {
        name: torch.full((batch, length), IGNORE_INDEX, dtype=torch.long)
        for name in _target_names(beam_levels, slur_slots)
    }

    for row, sequence in enumerate(sequences):
        for column, notation in enumerate(sequence):
            if notation is None:
                continue
            _fill_note(targets, row, column, notation, beam_levels, slur_slots)
    return targets


def _target_names(beam_levels: int, slur_slots: int) -> list[str]:
    names = [BEAM_HEAD.format(level=level) for level in range(1, beam_levels + 1)]
    names.append(STEM_HEAD)
    names.append(TIE_HEAD)
    names.append(DYNAMIC_HEAD)
    for slot in range(1, slur_slots + 1):
        names.append(SLUR_EVENT_HEAD.format(slot=slot))
        names.append(SLUR_SIDE_HEAD.format(slot=slot))
    return names


def _fill_note(
    targets: dict[str, torch.Tensor],
    row: int,
    column: int,
    notation: NoteNotation,
    beam_levels: int,
    slur_slots: int,
) -> None:
    for level in range(1, beam_levels + 1):
        state = notation.beam_levels[level - 1]
        if state == BeamLevelState.NOT_APPLICABLE:
            # The duration has fewer flags than this level, so there is no answer here -
            # not an answer of "not applicable".
            continue
        targets[BEAM_HEAD.format(level=level)][row, column] = _BEAM_INDEX[state]

    if notation.stem != StemDirection.UNKNOWN:
        targets[STEM_HEAD][row, column] = _STEM_INDEX[notation.stem]

    # Always supervised, like a slur event: most notes are not tied, and "no tie" is a
    # real prediction rather than an absence of one.
    targets[TIE_HEAD][row, column] = _TIE_INDEX[notation.tie]

    # Always supervised too: NONE (no dynamic attached) is 96.65% of notes (27.97) but is
    # a real prediction, the same shape of imbalance the tie head already handles.
    # Collapsed through trained_dynamic_mark, not indexed on notation.dynamic directly -
    # the head's own vocabulary is the phase16-measured trained subset (28.1), smaller
    # than the full representation NoteNotation carries.
    targets[DYNAMIC_HEAD][row, column] = _DYNAMIC_INDEX[trained_dynamic_mark(notation.dynamic)]

    for slot in range(1, slur_slots + 1):
        event, side = notation.slurs[slot - 1]
        targets[SLUR_EVENT_HEAD.format(slot=slot)][row, column] = _EVENT_INDEX[event]
        if side != SlurSide.UNSPECIFIED:
            targets[SLUR_SIDE_HEAD.format(slot=slot)][row, column] = _SIDE_INDEX[side]


def notation_positions(
    symbols: Sequence["EncodedSymbol"], length: int
) -> list[NoteNotation | None]:
    """Line notation up with the decoder's token positions.

    to_decoder_branches lays a sequence out as BOS, the symbols, EOS, then padding, and
    the targets must sit at exactly those indices or every label lands one place from the
    note it describes. Mirroring the layout here - rather than assuming the caller
    offsets correctly - keeps the two definitions together, and the test compares this
    against the real branch tensors so they cannot drift apart.

    BOS, EOS and padding carry no notation, and neither do symbols that are not notes.
    """
    positions: list[NoteNotation | None] = [None]
    positions.extend(symbol.notation for symbol in symbols)
    positions.append(None)
    if len(positions) < length:
        positions.extend([None] * (length - len(positions)))
    return positions[:length]


def align_to_decoder_output(targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Shift targets onto the decoder's output positions.

    The decoder is autoregressive: it reads `rhythms[:, :-1]` and its hidden state at
    position t is the prediction for token t+1. The structured heads sit on that same
    hidden state, so a target tensor laid out over the full token sequence is one place
    to the left of the logits it is meant to score.

    Getting this wrong does not raise. The shapes differ by one, which torch would catch,
    but only until something pads or truncates - and then every head trains on the token
    after the one it describes, converges to something plausible, and reports a healthy
    loss. Hence a named function with a test rather than a `[:, 1:]` at the call site.
    """
    return {name: tensor[:, 1:] for name, tensor in targets.items()}
