"""
Output-only heads for beams, stem direction, ties and structured slurs.

These read the shared decoder's hidden state and predict alongside the existing heads.
They are a separate module rather than more projections inside ScoreTransformerWrapper
for two reasons, both about not disturbing what already works.

The wrapper's forward returns a positional tuple that ONNX export and every inference
path index by position, so appending outputs there is the exact hazard the design warns
about: adding one head silently shifts every downstream output. The hidden state is
already the seventh element of that tuple, so these heads can hang off it and change
nothing.

And they are output-only. Nothing here feeds back into the next autoregressive step, so
the pretrained input path is untouched and the existing heads' logits stay bit-identical
while the core is frozen - which is the whole point of the first experiment: does the
representation already carry enough visual evidence to learn explicit beaming, stem
direction and richer slurs?

Head names are the stable typed identifiers the review contract uses (`beam.level.1`,
`stem.direction`, `slur.slot.1.event`), so the same strings key the logits dictionary,
the capability manifest and the ONNX outputs, and nothing has to translate between them.
"""

import torch
from torch import nn

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    DYNAMIC_CLASSES,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    TIE_CLASSES,
    TRAINED_BEAM_LEVELS,
    TRAINED_SLUR_SLOTS,
)

BEAM_HEAD = "beam.level.{level}"
STEM_HEAD = "stem.direction"
TIE_HEAD = "tie.state"
DYNAMIC_HEAD = "dynamic.mark"
SLUR_EVENT_HEAD = "slur.slot.{slot}.event"
SLUR_SIDE_HEAD = "slur.slot.{slot}.side"


def head_names(
    beam_levels: int = TRAINED_BEAM_LEVELS, slur_slots: int = TRAINED_SLUR_SLOTS
) -> list[str]:
    """Every head this configuration predicts, in a stable order.

    A consumer must treat a name absent from here as unsupported rather than as a
    confident prediction of nothing - which is why the manifest carries the list rather
    than callers assuming the full set.
    """
    names = [BEAM_HEAD.format(level=level) for level in range(1, beam_levels + 1)]
    names.append(STEM_HEAD)
    names.append(TIE_HEAD)
    names.append(DYNAMIC_HEAD)
    for slot in range(1, slur_slots + 1):
        names.append(SLUR_EVENT_HEAD.format(slot=slot))
        names.append(SLUR_SIDE_HEAD.format(slot=slot))
    return names


class StructuredNotationHeads(nn.Module):
    """Projections from the decoder hidden state to beam, stem and slur logits."""

    def __init__(
        self,
        dim: int,
        beam_levels: int = TRAINED_BEAM_LEVELS,
        slur_slots: int = TRAINED_SLUR_SLOTS,
    ) -> None:
        super().__init__()
        if beam_levels < 0 or slur_slots < 0:
            raise ValueError("beam_levels and slur_slots must not be negative")
        self.beam_levels = beam_levels
        self.slur_slots = slur_slots
        self.beam = nn.ModuleList(
            [nn.Linear(dim, len(BEAM_LEVEL_CLASSES)) for _ in range(beam_levels)]
        )
        self.stem = nn.Linear(dim, len(STEM_CLASSES))
        # A tie needs no slot, unlike a slur: it joins one pitch to the same pitch, so two
        # cannot be open on one note of a voice without being the same tie.
        self.tie = nn.Linear(dim, len(TIE_CLASSES))
        # Also no slot: a note carries at most one dynamic mark by construction (the
        # attachment rule claims the single pending mark per staff, 27.97/27.94).
        self.dynamic = nn.Linear(dim, len(DYNAMIC_CLASSES))
        self.slur_event = nn.ModuleList(
            [nn.Linear(dim, len(SLUR_EVENT_CLASSES)) for _ in range(slur_slots)]
        )
        self.slur_side = nn.ModuleList(
            [nn.Linear(dim, len(SLUR_SIDE_CLASSES)) for _ in range(slur_slots)]
        )

    def head_names(self) -> list[str]:
        return head_names(self.beam_levels, self.slur_slots)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map the decoder hidden state to one logits tensor per head.

        A dictionary rather than a tuple: these are the outputs most likely to grow, and
        a positional contract is what made adding a head risky in the first place.
        """
        logits: dict[str, torch.Tensor] = {}
        for index, projection in enumerate(self.beam):
            logits[BEAM_HEAD.format(level=index + 1)] = projection(hidden)
        logits[STEM_HEAD] = self.stem(hidden)
        logits[TIE_HEAD] = self.tie(hidden)
        logits[DYNAMIC_HEAD] = self.dynamic(hidden)
        for index in range(self.slur_slots):
            logits[SLUR_EVENT_HEAD.format(slot=index + 1)] = self.slur_event[index](hidden)
            logits[SLUR_SIDE_HEAD.format(slot=index + 1)] = self.slur_side[index](hidden)
        return logits
