"""Turn the structured heads' logits into notation, plus the choices a user may revise.

The heads emit one logits vector per head per note. Two different things are wanted from
those numbers and they must not be conflated:

* **What to write.** The argmax of each head, assembled into a `NoteNotation` that
  `music_xml_generator` turns into `<beam>`, `<stem>` and slur placement.
* **What to offer.** Where the model is genuinely uncertain, the ranked alternatives a
  user can pick between.

A head can be good enough for the first and not the second. Stems predict at macro F1
0.7189 (micro 0.9483) - worth writing, but presenting a stem as a *choice* implies the
model has an opinion worth arbitrating, and at that level the interface would be
overclaiming. `OFFERED_HEADS` encodes that distinction; see
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` for the per-head table it comes from.

Deliberately free of torch. The generation loop hands in plain sequences of floats, which
keeps this module testable without a GPU stack and keeps the ONNX path - where logits
arrive as numpy - from needing a second implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    DYNAMIC_CLASSES,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    TIE_CLASSES,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
)

BEAM_HEAD = "beam.level.{level}"
STEM_HEAD = "stem.direction"
TIE_HEAD = "tie.state"
DYNAMIC_HEAD = "dynamic.mark"
SLUR_EVENT_HEAD = "slur.slot.{slot}.event"
SLUR_SIDE_HEAD = "slur.slot.{slot}.side"

#: The head vocabularies, in the order the heads' logits are laid out.
HEAD_CLASSES: dict[str, tuple] = {
    STEM_HEAD: STEM_CLASSES,
    TIE_HEAD: TIE_CLASSES,
    DYNAMIC_HEAD: DYNAMIC_CLASSES,
}

#: Heads whose alternatives may be shown to a user, decided 2026-08-25.
#:
#: Beams and slurs only. Stems (0.7189) and ties (0.8032) are written to MusicXML but not
#: offered - see the module docstring. Beam level 4 has support 8, so its distribution is
#: noise and must never be rendered as a set of choices, however confident it looks.
OFFERED_HEAD_PREFIXES = ("beam.level.", "slur.slot.")
MAX_OFFERED_BEAM_LEVEL = 3

#: Below this, the top class is not treated as settled and alternatives are offered.
#: Provisional: the real value should come from the confidence distribution measured on
#: pages, which has not been done. Picking it from intuition is exactly the kind of
#: unmeasured constant this project has been bitten by, so it is a parameter everywhere
#: and this default is only a starting point.
DEFAULT_CONFIDENCE_THRESHOLD = 0.85

#: Dynamics are not emitted at all: macro F1 0.1030, the head never trained. Writing them
#: would put confident-looking wrong marks into the score.
EMIT_DYNAMICS = False


@dataclass(frozen=True)
class Alternative:
    """One class a head considered, with its probability."""

    value: str
    probability: float


@dataclass(frozen=True)
class HeadChoice:
    """A head's prediction and, when uncertain, what else it was weighing.

    `alternatives` is empty when the head is confident or when the head is not one users
    may revise. An empty list is the signal to show nothing - callers should not have to
    re-apply the policy.
    """

    head: str
    value: str
    probability: float
    alternatives: tuple[Alternative, ...] = ()

    @property
    def is_uncertain(self) -> bool:
        return bool(self.alternatives)


@dataclass(frozen=True)
class StructuredPrediction:
    """What one note's heads produced: the notation to write, and the choices to offer."""

    notation: NoteNotation
    choices: tuple[HeadChoice, ...] = ()

    def uncertain_choices(self) -> tuple[HeadChoice, ...]:
        return tuple(choice for choice in self.choices if choice.is_uncertain)


def softmax(logits: Sequence[float]) -> list[float]:
    """Probabilities from logits, shifted by the max so large values cannot overflow."""
    if not logits:
        return []
    largest = max(logits)
    exponentiated = [math.exp(value - largest) for value in logits]
    total = sum(exponentiated)
    return [value / total for value in exponentiated]


def is_offered(head: str) -> bool:
    """Whether this head's alternatives may be shown to a user.

    Beam level 4 is excluded by number rather than by name: it shares a prefix with the
    levels that are offered, and its support of 8 makes its distribution meaningless.
    """
    if not head.startswith(OFFERED_HEAD_PREFIXES):
        return False
    if head.startswith("beam.level."):
        try:
            return int(head.rsplit(".", 1)[1]) <= MAX_OFFERED_BEAM_LEVEL
        except ValueError:
            return False
    return True


def decode_head(
    head: str,
    logits: Sequence[float],
    classes: Sequence,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> HeadChoice:
    """One head's logits as a chosen value plus, when warranted, ranked alternatives."""
    probabilities = softmax(logits)
    ranked = sorted(
        zip(classes, probabilities, strict=False), key=lambda pair: pair[1], reverse=True
    )
    best, confidence = ranked[0]

    alternatives: tuple[Alternative, ...] = ()
    if is_offered(head) and confidence < threshold:
        # Every class is included, not only the runners-up: a user deciding between two
        # readings is better served by seeing that the third was near-zero than by having
        # it hidden. Ordering carries the emphasis.
        alternatives = tuple(
            Alternative(str(value), probability) for value, probability in ranked
        )

    return HeadChoice(head, str(best), confidence, alternatives)


def _classes_for(head: str) -> tuple:
    if head.startswith("beam.level."):
        return BEAM_LEVEL_CLASSES
    if head.endswith(".event"):
        return SLUR_EVENT_CLASSES
    if head.endswith(".side"):
        return SLUR_SIDE_CLASSES
    return HEAD_CLASSES[head]


def decode_note(
    logits: dict[str, Sequence[float]],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> StructuredPrediction:
    """Every head's logits for one note, as notation plus the choices worth offering.

    Missing heads fall back to their neutral state rather than raising: a checkpoint
    trained before a head existed must keep decoding, which is the same rule
    `configs.py` applies to enabling the heads at all.
    """
    choices = [
        decode_head(head, values, _classes_for(head), threshold)
        for head, values in sorted(logits.items())
        if head in HEAD_CLASSES or head.startswith(("beam.level.", "slur.slot."))
    ]
    by_head = {choice.head: choice for choice in choices}

    beam_levels = tuple(
        BeamLevelState(by_head[name].value)
        for level in range(1, 7)
        if (name := BEAM_HEAD.format(level=level)) in by_head
    )

    slots = sorted(
        {
            int(head.split(".")[2])
            for head in by_head
            if head.startswith("slur.slot.") and head.endswith(".event")
        }
    )
    slurs = tuple(
        (
            SlurEvent(by_head[SLUR_EVENT_HEAD.format(slot=slot)].value),
            SlurSide(side.value) if (side := by_head.get(SLUR_SIDE_HEAD.format(slot=slot))) else SlurSide.UNSPECIFIED,
        )
        for slot in slots
    )

    stem = (
        StemDirection(by_head[STEM_HEAD].value)
        if STEM_HEAD in by_head
        else StemDirection.NOT_APPLICABLE
    )
    tie = TieState(by_head[TIE_HEAD].value) if TIE_HEAD in by_head else TieState.NONE
    dynamic = (
        DynamicMark(by_head[DYNAMIC_HEAD].value)
        if EMIT_DYNAMICS and DYNAMIC_HEAD in by_head
        else DynamicMark.NONE
    )

    notation = NoteNotation(
        beam_levels=beam_levels, stem=stem, slurs=slurs, tie=tie, dynamic=dynamic
    )
    # Only offered heads can carry alternatives, so the full list is safe to hand on;
    # callers filter with uncertain_choices() rather than re-deriving the policy.
    return StructuredPrediction(notation=notation, choices=tuple(choices))
