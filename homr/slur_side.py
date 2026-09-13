"""Choose which side of the noteheads a slur sits on, between the head and a convention.

Engravers place a slur opposite the stems: stems up, slur below the noteheads; stems down,
slur above. That is a convention rather than a preference, which makes the side derivable
from something the pipeline already produces - and the slur-side head is the weakest of the
shipped heads, reported at macro-F1 .723 against no baseline at all.

Measured in place over 2,000 held-out scanned staves (`training/transformer/
end_to_end_passes.py`), on the 4,265 sides the engraving actually states:

    the trained slur-side head          78.80%
    opposite the arbitrated stem        89.27%     no parameters
    head when confident, else the rule  90.62%     a threshold

The rule is derived from the stem this pipeline *predicted*, after `stem_arbitration`, not
from the engraved stem - so it already carries every stem mistake. `slur_side_baseline.py`
reported 94.0% from engraved stems; five of those points are the cost of the chain, and
what is left still beats the head by eleven.

They fail on different notes, which is why this arbitrates rather than replaces: the rule
rescues 730 sides the head gets wrong and the head rescues 285 the rule gets wrong, with
only 172 defeating both. Most of the value is the rule; the threshold adds half a point on
top of it.

Runs after `arbitrate_stems`, and must: the stem it reads is the arbitrated one, and
deriving a side from a stem the pipeline is about to change would describe a score nobody
renders.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from homr.transformer.structured_decode import SLUR_SIDE_HEAD
from homr.transformer.structured_notation import SlurEvent, SlurSide, StemDirection
from homr.transformer.vocabulary import EncodedSymbol

#: The head is trusted at or above this. Swept on half the staves and reported on the
#: other half, so it is not fitted to the number it is quoted against: 90.62% at 0.9
#: against 90.11% for the rule alone and 78.60% for the head alone, on the reserved half.
HEAD_CONFIDENCE_THRESHOLD = 0.9

#: The convention. A note whose stem the pipeline could not determine has no entry here,
#: and the rule then has nothing to say.
OPPOSITE = {
    StemDirection.UP: SlurSide.BELOW,
    StemDirection.DOWN: SlurSide.ABOVE,
}


@dataclass
class SlurArbitration:
    """What the pass did, so a caller reports rather than infers."""

    sides: int = 0
    head_kept: int = 0
    rule_applied: int = 0
    changed: int = 0

    def describe(self) -> str:
        return (
            f"slur side: {self.sides:,} spans, head kept {self.head_kept:,}, "
            f"rule applied {self.rule_applied:,}, side changed {self.changed:,}"
        )


def _confidence(symbol: EncodedSymbol, slot: int) -> float | None:
    """The head's own probability for this slot's side, if the heads ran."""
    wanted = SLUR_SIDE_HEAD.format(slot=slot + 1)
    for choice in getattr(symbol, "structured_choices", ()) or ():
        if choice.head == wanted:
            return choice.probability
    return None


def choose_slur_sides(
    staff: Sequence[EncodedSymbol], threshold: float = HEAD_CONFIDENCE_THRESHOLD
) -> SlurArbitration:
    """Apply the opposite-the-stem convention where the head is unsure, in place.

    Only slots carrying an actual span are touched. A slot with no event has no side to
    place, and a note whose stem is unknown gives the rule nothing to derive from - in
    both cases the head's own answer stands, because an invented side is worse than an
    uncertain one.
    """
    report = SlurArbitration()
    for symbol in staff:
        notation = getattr(symbol, "notation", None)
        if notation is None or not notation.slurs:
            continue

        sides = list(notation.slurs)
        changed_here = False
        for slot, (event, side) in enumerate(sides):
            if event == SlurEvent.NONE:
                continue
            derived = OPPOSITE.get(notation.stem)
            if derived is None:
                continue
            report.sides += 1
            confidence = _confidence(symbol, slot)
            if confidence is not None and confidence >= threshold:
                report.head_kept += 1
                continue
            report.rule_applied += 1
            if side != derived:
                report.changed += 1
                sides[slot] = (event, derived)
                changed_here = True

        if changed_here:
            symbol.notation = replace(notation, slurs=tuple(sides))
    return report
