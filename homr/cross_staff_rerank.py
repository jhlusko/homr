"""
Phase 1: decode-time cross-staff-consistency reranking, no retraining
(`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2, `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §5).

The design calls this "k-best beam search for the rhythm head, scored by cross-staff
agreement." Built here as a narrower, cheaper mechanism that targets exactly the same
failure type (Type 1: one mis-decoded note that was a narrow, close call against a
reliable majority) without new batched-cache beam-search machinery: at each staff's
narrowest-margin rhythm decisions (`ScoreDecoder.generate_with_rhythm_margins`), branch
into a full alternate decode (`ScoreDecoder.rhythm_alternative`, built on
`generate_from_prefix` - already validated against the live model), then keep whichever
candidate - the original greedy decode, or one of its forks - best matches the *other*
staves' cumulative barline positions. This is deliberately not textbook fixed-width beam
search maintaining k hypotheses at every step: real classical beam search would need
per-step multi-hypothesis KV-cache batching this codebase has never exercised (batch=1
throughout `decoder_inference.py`), a much larger and riskier undertaking than the
targeted "branch only where the model was genuinely unsure" approach here, for the same
stated goal (§7.2's own criterion: "the greedy path's small local error is only narrowly
more likely than a nearby k-best candidate that also satisfies cross-staff agreement").

Deliberately does not fix Type 2 (systematic misread - not a narrow-margin call) or
Type 3 (chaotic disagreement) - see this module's own benchmark results for whether
that prediction holds.
"""
from collections import Counter
from typing import Any

from homr.cross_staff_consistency import _cumulative_barline_positions
from homr.transformer.decoder_inference import ScoreDecoder
from homr.transformer.vocabulary import EncodedSymbol


def fork_candidates_from_margins(
    decoder: ScoreDecoder,
    greedy: list[EncodedSymbol],
    margins: list[tuple[int, float]],
    max_forks: int = 3,
    **kwargs: Any,
) -> list[list[EncodedSymbol]]:
    """The expensive half of candidate generation, split out from
    `rhythm_candidates_for_staff` so a caller that already has `greedy`/`margins`
    cheaply in hand (`generate_with_rhythm_margins` costs the same as a plain
    `generate()` call - only forking is expensive, one full extra `generate_from_prefix`
    decode per fork) can gate this behind a cheap check instead of always paying for
    `max_forks` extra full decodes on every staff, most of which end up unused on a
    page with no cross-staff disagreement at all (`parse_staffs` does exactly this:
    only forks a system's staves once Stage A already shows a finding on their greedy
    decode).

    Returns the greedy decode plus up to `max_forks` alternates branched at the
    narrowest rhythm-decision margins.
    """
    candidates = [greedy]
    if not margins:
        return candidates

    forkable = sorted(range(len(margins)), key=lambda i: margins[i][1])
    for step in forkable[:max_forks]:
        alt_token_id, _margin = margins[step]
        candidates.append(decoder.rhythm_alternative(greedy, step, alt_token_id, **kwargs))
    return candidates


def rhythm_candidates_for_staff(
    decoder: ScoreDecoder,
    start_tokens: Any,
    nonote_tokens: Any,
    max_forks: int = 3,
    **kwargs: Any,
) -> list[list[EncodedSymbol]]:
    """One staff's greedy decode plus up to `max_forks` alternate decodes, branched at
    its `max_forks` narrowest rhythm-decision margins (`generate_with_rhythm_margins`).
    `candidates[0]` is always the plain greedy decode - the reranking step below falls
    back to it if no alternative helps. Steps within 1 of the sequence's end are not
    forked (`rhythm_alternative` needs at least the corrected token itself to regenerate
    a continuation from; nothing meaningful to gain forking the very last token either).

    Unconditionally forks - callers that want to gate the expensive forking step behind
    a cheap check (as `parse_staffs` does) should call `generate_with_rhythm_margins`
    and `fork_candidates_from_margins` directly instead.
    """
    greedy, margins, _hidden = decoder.generate_with_rhythm_margins(
        start_tokens, nonote_tokens, **kwargs
    )
    return fork_candidates_from_margins(decoder, greedy, margins, max_forks=max_forks, **kwargs)


def rerank_staff_candidates(
    candidates_by_staff: dict[int, list[list[EncodedSymbol]]],
    min_corroborating_staves: int = 2,
) -> dict[int, list[EncodedSymbol]]:
    """For each staff with alternate candidates, keep whichever candidate - the
    original greedy decode or one of its forks - has cumulative barline positions
    (`_cumulative_barline_positions`, the same signal `check_barline_positions`/
    `propose_majority_position_corrections` already use) agreeing most with the
    *majority position at each barline among the other staves*. A staff is never
    compared against its own candidates when computing that majority (which would be
    circular), and reranking is skipped entirely for a staff with fewer than
    `min_corroborating_staves` other staves reporting a barline at all - the same "don't
    guess without real corroboration" bar `propose_majority_position_corrections` uses
    (that function's own default is a 3-staff majority; this defaults to requiring 2
    *other* staves, i.e. 3 total, matching it).

    Every staff's greedy decode is always the baseline choice unless a specific
    alternative agrees with the majority strictly more often than the greedy decode
    does - never picks a worse-or-equal alternative over the greedy default.
    """
    greedy = {i: c[0] for i, c in candidates_by_staff.items()}
    final = dict(greedy)

    for staff_index, candidates in candidates_by_staff.items():
        if len(candidates) < 2:
            continue  # nothing to rerank against

        other_positions = [
            _cumulative_barline_positions(final[j]) for j in final if j != staff_index
        ]
        other_positions = [p for p in other_positions if p]
        if len(other_positions) < min_corroborating_staves:
            continue

        shortest = min(len(p) for p in other_positions)
        if shortest == 0:
            continue
        majority = [
            Counter(p[idx] for p in other_positions).most_common(1)[0][0]
            for idx in range(shortest)
        ]

        def agreement(candidate: list[EncodedSymbol], majority: list = majority) -> int:
            positions = _cumulative_barline_positions(candidate)
            n = min(len(positions), len(majority))
            return sum(1 for k in range(n) if positions[k] == majority[k])

        best = max(candidates, key=agreement)
        if agreement(best) > agreement(candidates[0]):
            final[staff_index] = best

    return final
