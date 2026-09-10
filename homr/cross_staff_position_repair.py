"""Turn a localized barline-position divergence into a *verified* correction.

`cross_staff_repair.propose_majority_position_corrections` says a staff's cumulative
barline positions diverge from a majority of its siblings by a constant offset from one
measure onward. Its own docstring explains why it has no `apply_*` counterpart: the
signature says *where* the divergence starts, not *what* to change - which note or rest,
and how. Guessing at that is exactly what this codebase refuses to do.

This module does not guess. It uses the proposal as a *search hint* rather than a
correction: the divergence localizes the fault to one measure, so re-decode only the
rhythm decisions inside that measure, and accept an alternative only when its barline
positions match the majority's **exactly**. A candidate that merely agrees better is
rejected. The acceptance test is the verification, which is what lets this apply
automatically where the proposal alone could not.

Why this is not already covered by Phase 1's reranker: that forks a staff's
`max_forks` narrowest margins across the *whole* staff and keeps whichever candidate
agrees *most*. A divergence surviving it is one whose causal decision was not among
those few narrowest margins - so this searches a much smaller region (one measure) far
more thoroughly, and demands exactness rather than improvement.

The decoder is injected as `fork`, so every rule here is testable without one.
"""

from collections.abc import Callable, Sequence
from collections import Counter
from fractions import Fraction

from homr.cross_staff_consistency import _cumulative_barline_positions
from homr.cross_staff_repair import propose_majority_position_corrections
from homr.transformer.vocabulary import EncodedSymbol

#: How many rhythm decisions inside the divergent measure to try. Higher than Phase 1's
#: 3 because the search space is one measure rather than a whole staff, and every
#: candidate is verified before use - a wrong one costs a decode, not a wrong bar.
DEFAULT_MAX_FORKS = 6


def _is_barline(symbol: EncodedSymbol) -> bool:
    """Matches `music_xml_generator.SymbolChord.is_barline`, at symbol level."""
    return "barline" in symbol.rhythm or "repeat" in symbol.rhythm


def barline_count(symbols: Sequence[EncodedSymbol]) -> int:
    return sum(1 for symbol in symbols if _is_barline(symbol))


def measure_span(symbols: Sequence[EncodedSymbol], measure_index: int) -> tuple[int, int] | None:
    """`[start, end)` symbol indices of the `measure_index`-th barline-delimited measure.

    Measure 0 runs from the start to the first barline. None when the staff does not
    have that many measures - a proposal built against a different sequence than the one
    handed here is not something to apply blindly.
    """
    if measure_index < 0:
        return None
    start = 0
    seen = 0
    for index, symbol in enumerate(symbols):
        if not _is_barline(symbol):
            continue
        if seen == measure_index:
            return (start, index)
        seen += 1
        start = index + 1
    return None


def majority_barline_sequence(
    staves: Sequence[Sequence[EncodedSymbol]], min_corroborating: int = 3
) -> tuple[Fraction, ...] | None:
    """The cumulative barline sequence a clear majority of these staves agree on.

    Mirrors `propose_majority_position_corrections`' own bar deliberately: at least
    `min_corroborating` staves agreeing exactly, and no tie. A repair accepted against a
    weaker majority than the one that proposed it would be applying a correction the
    proposer would have declined to make.
    """
    positions = {
        index: _cumulative_barline_positions(list(staff)) for index, staff in enumerate(staves)
    }
    with_barlines = {index: value for index, value in positions.items() if value}
    if len(with_barlines) < 4:
        return None
    shortest = min(len(value) for value in with_barlines.values())
    if shortest == 0:
        return None
    truncated = [tuple(value[:shortest]) for value in with_barlines.values()]
    counts = Counter(truncated)
    majority, count = counts.most_common(1)[0]
    if count < min_corroborating:
        return None
    if list(counts.values()).count(count) > 1:
        return None  # a genuine tie - the same refusal the proposer makes
    return majority


def agrees_exactly(
    candidate: Sequence[EncodedSymbol], majority: Sequence[Fraction]
) -> bool:
    """Whether this candidate's barlines land exactly where the majority's do.

    Compared over the majority's length: a candidate that reproduces every barline
    position the majority states, and then continues, has resolved the divergence. One
    that stops short has not - it no longer disagrees only because there is less of it.
    """
    positions = _cumulative_barline_positions(list(candidate))
    if len(positions) < len(majority):
        return False
    return tuple(positions[: len(majority)]) == tuple(majority)


def repair_position_divergence(
    staves: Sequence[Sequence[EncodedSymbol]],
    raw_staves: Sequence[Sequence[EncodedSymbol]],
    margins_by_staff: Sequence[Sequence[tuple[int, float]]],
    fork: Callable[[int, int, int], list[EncodedSymbol] | None],
    max_forks: int = DEFAULT_MAX_FORKS,
) -> dict[int, list[EncodedSymbol]]:
    """Corrected staves, keyed by staff index. Absent means nothing verified.

    `staves` are the filtered per-staff decodes this system's findings are computed
    from; `raw_staves` and `margins_by_staff` are the unfiltered sequences the decoder
    numbers its steps against (`margins[i]` belongs to `raw[i]`, aligned 1:1 -
    `generate_with_rhythm_margins` guarantees this). `fork(staff_index, step,
    alt_token_id)` returns a filtered candidate, or None if that fork failed.
    """
    proposals = propose_majority_position_corrections(list(staves))
    if not proposals:
        return {}
    majority = majority_barline_sequence(staves)
    if majority is None:
        return {}

    repaired: dict[int, list[EncodedSymbol]] = {}
    for proposal in proposals:
        index = proposal.staff_index
        if index >= len(raw_staves) or index >= len(margins_by_staff):
            continue
        raw = raw_staves[index]
        margins = margins_by_staff[index]
        if len(margins) != len(raw):
            continue  # not the 1:1 alignment the step numbering depends on
        # The proposal counts measures in the filtered staff; the steps to fork are
        # numbered against the raw one. Filtering drops lower-staff symbols, never
        # barlines - but if that ever stops being true the measure would be the wrong
        # one, so require the two to agree on how many measures exist rather than
        # assume it.
        if barline_count(raw) != barline_count(staves[index]):
            continue
        span = measure_span(raw, proposal.measure_index)
        if span is None:
            continue
        steps = [step for step in range(*span) if step < len(margins)]
        steps.sort(key=lambda step: margins[step][1])
        for step in steps[:max_forks]:
            candidate = fork(index, step, margins[step][0])
            if candidate and agrees_exactly(candidate, majority):
                repaired[index] = list(candidate)
                break
    return repaired
