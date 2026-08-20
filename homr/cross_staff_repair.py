"""
Cross-staff context and repair, Stage B tier 1 (design §12.2, refined per
ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §4): propose - never apply - a deterministic
correction when a system's staves disagree on their opening key or time signature, when
a staff is silently missing a key signature every other staff agrees on, or when three
or more staves corroborate the same motif and one of them disagrees on a note's
articulation.

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

import difflib
from collections import Counter
from dataclasses import dataclass

from homr.transformer.vocabulary import EncodedSymbol

#: Matches check_shared_motifs' own default (homr/cross_staff_consistency.py) - both
#: exist to answer the same underlying question and should agree on what counts as "a
#: motif," even though this module never imports that one (staying decoupled the same
#: way tier 1's key/time-signature repair stays independent of the Finding it repairs).
_DEFAULT_MIN_MOTIF_LENGTH = 4


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


@dataclass(frozen=True)
class InsertionRepairProposal:
    """One proposed token *insertion*, on one staff, before one position - unlike
    `RepairProposal` (replaces an existing token) or `ArticulationRepairProposal`
    (replaces one field of an existing token), there is nothing at `position` to
    replace here: the staff never decoded a key signature at all, so a symbol is added,
    shifting every later position on that staff by one. `apply_proposal`/
    `apply_articulation_proposal` cannot express that, hence a dedicated type and a
    dedicated `apply_insertion_proposal`.
    """

    staff_index: int
    #: Insert *before* this index - 0 inserts at the very start of the staff.
    position: int
    inserted_rhythm: str
    reason: str


def propose_carry_forward_key_signature(
    staves: list[list[EncodedSymbol]],
) -> list[InsertionRepairProposal]:
    """A staff with *no* key signature token anywhere in this system's decode, when
    every other staff that states one agrees on the same value - modern engraving
    convention restates the key signature on every staff at every system (confirmed
    directly against a real page's ground-truth MusicXML this session: all four parts
    of a string quartet system carried an identical `<key>` element the model decoded
    correctly on only one of the four), so a silent staff here reads as a decode
    omission, not a genuine "this part carries no key signature."

    Deliberately does **not** generalise to time signature: that field's own convention
    is the opposite (restated only when it changes), so a silent staff there is normal,
    expected behaviour, not evidence of anything missing - applying this same carry-
    forward logic to time signature would be a real, wrong guess, not a small extension.

    Fires whenever at least one staff states a key signature and no *other* staff states
    a genuinely conflicting value - deliberately a lower bar than
    `propose_majority_correction`'s "at least two staves stating a value," since a lone
    stated value with everyone else silent is exactly the case this convention argument
    covers; two conflicting stated values is not this rule's territory at all, and is
    left entirely to `propose_majority_correction` instead of guessing between the two
    mechanisms for the same system.
    """
    stated = [
        (index, *result)
        for index, staff in enumerate(staves)
        if (result := _first_of_prefix(staff, "keySignature")) is not None
    ]
    if not stated:
        return []
    values = {rhythm for _, _, rhythm in stated}
    if len(values) != 1:
        return []  # a genuine disagreement - propose_majority_correction's territory
    majority_value = next(iter(values))
    stating_indices = {index for index, _, _ in stated}
    silent_indices = [index for index in range(len(staves)) if index not in stating_indices]

    proposals = []
    for staff_index in silent_indices:
        staff = staves[staff_index]
        insert_at = 1 if staff and staff[0].rhythm.startswith("clef") else 0
        proposals.append(
            InsertionRepairProposal(
                staff_index=staff_index,
                position=insert_at,
                inserted_rhythm=majority_value,
                reason=(
                    f"staff {staff_index} states no key signature this system; every "
                    f"other staff that states one agrees on {majority_value!r} - modern "
                    "convention restates the key signature on every staff, so this "
                    "reads as a decode omission, not a genuine absence"
                ),
            )
        )
    return proposals


def apply_insertion_proposal(
    staff: list[EncodedSymbol], proposal: InsertionRepairProposal
) -> list[EncodedSymbol]:
    """A new symbol list with `proposal`'s symbol inserted before `proposal.position` -
    `staff` is never mutated in place, the same discipline every other `apply_*`
    function in this module uses.
    """
    if not 0 <= proposal.position <= len(staff):
        raise ValueError(
            f"position {proposal.position} out of range for a {len(staff)}-symbol staff"
        )
    inserted = EncodedSymbol(rhythm=proposal.inserted_rhythm)
    return staff[: proposal.position] + [inserted] + staff[proposal.position :]


def propose_repairs(staves: list[list[EncodedSymbol]]) -> list[RepairProposal]:
    """Every tier-1 proposal this module currently knows how to make for one system's
    staves - both the checks `cross_staff_consistency.analyze_system` can flag
    (`key_signature_mismatch`, `time_signature_mismatch`) that this module has a
    corresponding repair for. Does not consult `Finding`s directly: each
    `propose_majority_correction` call already returns nothing when there is no
    disagreement to propose against, the same as `analyze_system`'s own checks report
    nothing when staves agree - so recomputing here rather than gating on a `Finding`
    kind string keeps the two independent computations from silently drifting apart
    about what "a mismatch" means.
    """
    return propose_majority_correction(staves, "keySignature") + propose_majority_correction(
        staves, "timeSignature"
    )


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


@dataclass(frozen=True)
class ArticulationRepairProposal:
    """One proposed articulation replacement, on one staff, at one note - a sibling of
    `RepairProposal` rather than a reuse of it: that dataclass's `current_rhythm`/
    `proposed_rhythm` fields name a rhythm-token swap specifically, and
    `apply_proposal` only ever rewrites `rhythm`. An articulation correction changes a
    different field entirely, so it gets its own proposal type and its own
    `apply_articulation_proposal`, rather than overloading fields whose names would
    then lie about what they hold.
    """

    staff_index: int
    position: int
    current_articulation: str
    proposed_articulation: str
    reason: str


def _note_indices(symbols: list[EncodedSymbol]) -> list[int]:
    return [index for index, symbol in enumerate(symbols) if symbol.rhythm.startswith("note")]


def _pairwise_note_alignment(
    reference: list[EncodedSymbol], other: list[EncodedSymbol], min_motif_length: int
) -> dict[int, int]:
    """`reference`'s note positions covered by a matching (rhythm, pitch) run of at
    least `min_motif_length` against `other`, mapped to `other`'s corresponding
    position - the same matching `homr.cross_staff_consistency.check_shared_motifs`
    does, but returning the alignment itself rather than only the mismatches found
    within it, since finding a *majority* needs to know every staff that corroborates a
    position, not just the two staves in one pairwise comparison.
    """
    reference_notes = _note_indices(reference)
    other_notes = _note_indices(other)
    reference_keys = [(reference[k].rhythm, reference[k].pitch) for k in reference_notes]
    other_keys = [(other[k].rhythm, other[k].pitch) for k in other_notes]
    matcher = difflib.SequenceMatcher(a=reference_keys, b=other_keys, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        if block.size < min_motif_length:
            continue
        for offset in range(block.size):
            mapping[reference_notes[block.a + offset]] = other_notes[block.b + offset]
    return mapping


def propose_motif_articulation_corrections(
    staves: list[list[EncodedSymbol]], min_motif_length: int = _DEFAULT_MIN_MOTIF_LENGTH
) -> list[ArticulationRepairProposal]:
    """A majority-corroborated articulation correction, for every note where three or
    more staves independently play the same matching motif and do not all agree on its
    articulation.

    `homr.cross_staff_consistency.check_shared_motifs` only ever compares staves
    *pairwise* - a finding names two staves with no third staff's evidence attached,
    which is not enough to know which of the two misread (see that module's own
    docstring, and ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md §4's recorded reasoning for why
    a repair was not built directly on it). This answers the harder question that
    unblocks one: for one staff's note (the "reference"), every *other* staff is
    checked pairwise via `_pairwise_note_alignment`; if the note's position is covered
    by a matching run against at least two other staves, that is three staves total
    (the reference plus two corroborators) - a genuine majority to propose toward, the
    same "report nothing rather than guess" bar `propose_majority_correction` already
    holds key/time signature to.

    Deliberately anchored to one fixed reference staff per corroboration, never merged
    transitively across different reference staves: two *separate* real occurrences of
    a common short motif (e.g. four repeated quarter notes appearing at two unrelated
    points in a piece) must never be linked into one artificial "corroborating group"
    just because each occurrence happens to pairwise-match a different staff -
    transitive merging was considered and rejected specifically because it cannot rule
    that out. Only the lowest-indexed staff in a corroborating group is used as its
    reference, so each real group is reported exactly once rather than once per member.
    """
    proposals = []
    for reference_index in range(len(staves)):
        alignments = {
            other_index: _pairwise_note_alignment(
                staves[reference_index], staves[other_index], min_motif_length
            )
            for other_index in range(len(staves))
            if other_index != reference_index
        }
        for note_index in _note_indices(staves[reference_index]):
            group = [(reference_index, note_index)]
            for other_index, mapping in alignments.items():
                if note_index in mapping:
                    group.append((other_index, mapping[note_index]))
            if len(group) < 3:
                continue
            if reference_index != min(staff_index for staff_index, _ in group):
                continue  # reported from the lowest-indexed member only

            articulations = [staves[staff_index][pos].articulation for staff_index, pos in group]
            counts = Counter(articulations)
            majority_value, majority_count = counts.most_common(1)[0]
            if list(counts.values()).count(majority_count) > 1:
                continue  # a genuine tie - the same refusal propose_majority_correction uses

            for staff_index, pos in group:
                current = staves[staff_index][pos].articulation
                if current != majority_value:
                    proposals.append(
                        ArticulationRepairProposal(
                            staff_index=staff_index,
                            position=pos,
                            current_articulation=current,
                            proposed_articulation=majority_value,
                            reason=(
                                f"staff {staff_index} plays {current!r} at a note "
                                f"{majority_count}/{len(group)} corroborating staves "
                                f"play {majority_value!r}"
                            ),
                        )
                    )
    return proposals


def apply_articulation_proposal(
    staff: list[EncodedSymbol], proposal: ArticulationRepairProposal
) -> list[EncodedSymbol]:
    """A new symbol list with `proposal` applied - `staff` is never mutated in place,
    the same discipline `apply_proposal` uses for a rhythm-token swap.
    """
    if not 0 <= proposal.position < len(staff):
        raise ValueError(
            f"position {proposal.position} out of range for a {len(staff)}-symbol staff"
        )
    original = staff[proposal.position]
    if original.articulation != proposal.current_articulation:
        raise ValueError(
            f"staff at position {proposal.position} has articulation "
            f"{original.articulation!r}, not the {proposal.current_articulation!r} this "
            "proposal was built against - the staff has changed since the proposal was made"
        )
    corrected = EncodedSymbol(
        rhythm=original.rhythm,
        pitch=original.pitch,
        lift=original.lift,
        articulation=proposal.proposed_articulation,
        slur=original.slur,
        position=original.position,
        coordinates=original.coordinates,
        notation=original.notation,
    )
    return staff[: proposal.position] + [corrected] + staff[proposal.position + 1 :]
