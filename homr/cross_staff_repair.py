"""
Cross-staff context and repair, Stage B tier 1 (design §12.2, refined per
ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §4): propose - never apply - a deterministic
correction when a system's staves disagree on their opening key or time signature.

Tier 1, not tier 2: 20 real forced-prefix conditioning tests
(`homr.transformer.decoder_inference.ScoreDecoder.generate_from_prefix`, recorded in
ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §4) found this model's future predictions do not
depend on prior token identity - a re-decode under a corrected key/time signature
reproduces the same continuation a plain token swap would, on every field tested. Given
that, a deterministic edit is not a worse repair than a model-in-the-loop one here, and
it is free.

Scoped to each staff's *opening* key/time signature specifically - the case §12.2's own
worked example is (a viola's measure reading 7/8 against every other part's 4/4), not a
full resequencing of every later key/time change in a staff's decode.
`cross_staff_consistency.check_key_signatures`/`check_time_signatures` compare the whole
sequence and can find a real disagreement deeper in a staff's own decode; proposing a
majority correction for *that* needs a different kind of alignment (which later change
corresponds to which) this module does not attempt.

Never touches MusicXML or a decoded symbol list in place - `propose_majority_correction`
returns proposals a caller logs, hands to a human reviewer, or passes to
`apply_proposal` explicitly. The same discipline §12.2 states for Stage B generally: "a
review question, not an automatic correction."
"""

from collections import Counter
from dataclasses import dataclass

from homr.transformer.vocabulary import EncodedSymbol


@dataclass(frozen=True)
class RepairProposal:
    """One proposed token replacement, on one staff, at one position - never applied
    until a caller passes it to `apply_proposal`."""

    staff_index: int
    position: int
    current_rhythm: str
    proposed_rhythm: str
    reason: str


def _first_of_prefix(symbols: list[EncodedSymbol], prefix: str) -> tuple[int, str] | None:
    for index, symbol in enumerate(symbols):
        if symbol.rhythm.startswith(prefix):
            return index, symbol.rhythm
    return None


def propose_majority_correction(
    staves: list[list[EncodedSymbol]], prefix: str
) -> list[RepairProposal]:
    """One proposal per staff whose *opening* `prefix`-tagged token (`"keySignature"` or
    `"timeSignature"`) disagrees with the majority.

    None where there is no majority to propose - a genuine tie, or fewer than two
    staves stating one at all - the same "report nothing rather than guess" discipline
    `score_profile_layout.py` already uses: a repair with no clear majority is exactly
    the case a human reviewer belongs in, not a silent pick between equally-supported
    values.
    """
    found = [
        (index, *result)
        for index, staff in enumerate(staves)
        if (result := _first_of_prefix(staff, prefix)) is not None
    ]
    if len(found) < 2:
        return []

    values = [rhythm for _, _, rhythm in found]
    counts = Counter(values)
    majority_value, majority_count = counts.most_common(1)[0]
    if list(counts.values()).count(majority_count) > 1:
        return []

    return [
        RepairProposal(
            staff_index=staff_index,
            position=position,
            current_rhythm=rhythm,
            proposed_rhythm=majority_value,
            reason=(
                f"staff {staff_index} opens with {rhythm!r}; "
                f"{majority_count}/{len(found)} staves in this system open with "
                f"{majority_value!r}"
            ),
        )
        for staff_index, position, rhythm in found
        if rhythm != majority_value
    ]


def apply_proposal(staff: list[EncodedSymbol], proposal: RepairProposal) -> list[EncodedSymbol]:
    """A new symbol list with `proposal` applied - `staff` is never mutated in place, so
    a caller can compare before/after, or simply discard the proposal, without
    consequence.
    """
    if not 0 <= proposal.position < len(staff):
        raise ValueError(
            f"position {proposal.position} out of range for a {len(staff)}-symbol staff"
        )
    original = staff[proposal.position]
    if original.rhythm != proposal.current_rhythm:
        raise ValueError(
            f"staff at position {proposal.position} is {original.rhythm!r}, not the "
            f"{proposal.current_rhythm!r} this proposal was built against - the staff "
            "has changed since the proposal was made"
        )
    corrected = EncodedSymbol(
        rhythm=proposal.proposed_rhythm,
        pitch=original.pitch,
        lift=original.lift,
        articulation=original.articulation,
        slur=original.slur,
        position=original.position,
        coordinates=original.coordinates,
        notation=original.notation,
    )
    return staff[: proposal.position] + [corrected] + staff[proposal.position + 1 :]
