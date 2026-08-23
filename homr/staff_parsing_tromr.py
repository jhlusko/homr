from homr.model import Staff
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import ScoreDecoder
from homr.transformer.staff2score import Staff2Score
from homr.transformer.vocabulary import EncodedSymbol
from homr.type_definitions import NDArray

inference: Staff2Score | None = None


def parse_staff_tromr(staff: Staff, staff_image: NDArray, config: Config) -> list[EncodedSymbol]:
    return predict_best(staff_image, staff=staff, config=config)


def predict_best(org_image: NDArray, staff: Staff, config: Config) -> list[EncodedSymbol]:
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(config)

    result = inference.predict(org_image)
    if staff.is_grandstaff:
        return result
    return [r for r in result if r.position != "lower"]


def parse_staff_tromr_candidates(
    staff: Staff, staff_image: NDArray, config: Config, max_forks: int = 3
) -> list[list[EncodedSymbol]]:
    """Phase 1 (`DECODER_RHYTHM_ACCURACY_DESIGN.md` §7.2) counterpart to
    `parse_staff_tromr`: returns every candidate decode (greedy plus forks) instead of
    committing to one, for `parse_staffs` to rerank once a whole system's staves are
    available. `candidates[0]` is always what `parse_staff_tromr` alone would have
    returned - the same grandstaff/`position != "lower"` filter is applied to every
    candidate, not just the one eventually chosen, so a caller comparing candidates
    (cumulative barline positions, etc.) never sees the filtered-out lower-staff
    duplicate content any of them would otherwise carry."""
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(config)

    candidates = inference.predict_candidates(staff_image, max_forks=max_forks)
    if staff.is_grandstaff:
        return candidates
    return [[r for r in candidate if r.position != "lower"] for candidate in candidates]


def parse_staff_tromr_greedy_with_margins(
    staff: Staff, staff_image: NDArray, config: Config, staff_context_emb: NDArray | None = None
) -> tuple[
    list[EncodedSymbol], list[EncodedSymbol], list[tuple[int, float]], object, ScoreDecoder, NDArray
]:
    """Cheap first pass for Phase 1's live-pipeline gating (`parse_staffs`): costs
    exactly what a plain decode already costs - forking, the expensive part, is
    deferred until a caller decides (via a cheap Stage A check on the *filtered*
    greedy decode this returns) that it's actually worth it for this staff's system.

    Returns `(filtered_greedy, raw_greedy, margins, context, decoder, hidden_states)`:
    `filtered_greedy` is what `parse_staff_tromr` alone would have returned
    (grandstaff/`position != "lower"` applied); `raw_greedy`/`margins` are unfiltered
    and step-index-aligned, exactly what `fork_candidates_from_margins`/
    `rhythm_alternative` need to fork from later without losing that alignment;
    `context`/`decoder` let a later fork pass skip re-running the encoder;
    `hidden_states` is §4/§7.4 Stage C's own first-pass input (unused by Phase 1's own
    reranking, which this function also serves).

    `staff_context_emb`, given, makes this the *second* Stage C pass instead of the
    first - threaded straight through to the decoder, falling back to its own zero
    vector (today's single-pass behavior, unchanged) when omitted.
    """
    global inference  # noqa: PLW0603
    if inference is None:
        inference = Staff2Score(config)

    raw_greedy, margins, context, hidden_states = inference.predict_greedy_with_margins(
        staff_image, staff_context_emb=staff_context_emb
    )
    filtered_greedy = (
        raw_greedy if staff.is_grandstaff else [r for r in raw_greedy if r.position != "lower"]
    )
    return filtered_greedy, raw_greedy, margins, context, inference.decoder, hidden_states
