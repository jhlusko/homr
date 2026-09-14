"""Choose which side of the noteheads a slur sits on, between the head and a convention.

**This pass is off by default.** `HOMR_SLUR_SIDE=1` enables it. Read the scope below
before turning it on.

Engravers place a slur opposite the stems: stems up, slur below the noteheads; stems down,
slur above. Measured against engraved placements, that convention holds on one corpus and
does not hold at all on another:

    OSSQ (string quartet)       6,026 scorable   94.8%   macro-F1 .947
    Lieder v8 (voice + piano)   1,600 scorable   51.3%   macro-F1 .500

Chance on the second. The obvious explanation - that a polyphonic keyboard staff uses the
stem to encode which *voice* a note belongs to, not which side its slur sits on - is not
sufficient, because Lieder's own single-staff, one-voice slice reads 59.5%. The split is
by repertoire, not by any structural property a page carries, so **there is no gate that
can be applied at inference to tell the two cases apart.**

The pass was shipped on the strength of the first corpus alone: measured in place over
2,000 held-out *scanned OSSQ* staves, the head alone reached 78.80%, the rule alone 89.27%
and the arbitration 90.62%, tuned on half the staves and reported on the other. That +10.2
points is real and is OSSQ-scoped. Applied to piano material the same pass overwrites the
head with a coin flip wherever the head's confidence falls below the threshold.

One further caution before anyone re-validates it. OSSQ's engraved sides may record
MuseScore's own default placement rather than an engraver's decision, in which case the
94.8% is partly circular - the rule would be reproducing the layout algorithm that wrote
the labels. That was not settled here.

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
