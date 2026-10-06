"""Diagnostic vocal-part and leading-measure context without native references."""


def vocal_voice_candidates(final, transforms, routing):
    """Require one text-supported, exclusively five-line runtime voice.

    Physical text rows are evidence, not certified semantic lyric roles. Ambiguous
    candidates fail closed; missing-provenance notes never establish staff identity.
    """
    by_call = {t["staff_call"]: t for t in transforms}
    supported = {r["staff_call"] for r in routing["routed"] if "physical_row_order" in r}
    records = []
    for voice in sorted({r["voice_index"] for r in final}):
        calls = {
            r["staff_call"]
            for r in final
            if r["voice_index"] == voice and r.get("staff_call") is not None
        }
        single = bool(calls) and all(
            call in by_call
            and by_call[call]["staff_grid"]
            and all(len(point["y"]) == 5 for point in by_call[call]["staff_grid"])
            for call in calls
        )
        records.append(
            {
                "voice_index": voice,
                "staff_calls": sorted(calls),
                "five_line_only": single,
                "text_supported_calls": sorted(calls & supported),
            }
        )
    return records


def select_vocal_voice(final, transforms, routing):
    records = vocal_voice_candidates(final, transforms, routing)
    eligible = [r for r in records if r["five_line_only"] and r["text_supported_calls"]]
    basis = "unique_text_supported_five_line_voice"
    if len(eligible) > 1:
        recurring = [r for r in eligible if len(r["text_supported_calls"]) >= 2]
        if len(recurring) == 1:
            eligible = recurring
            basis = "unique_recurring_text_support_across_prepared_staffs"
    if len(eligible) != 1:
        raise ValueError("No unique text-supported five-line vocal voice")
    return {
        "voice_index": eligible[0]["voice_index"],
        "candidates": records,
        "selection_basis": basis,
        "semantic_lyric_role_certified": False,
        "native_input_used": False,
    }


def leading_measure_context(voice, final, transforms, trace, exported_by_part):
    """Infer a prefix only from completely preceding grand staffs and traced XML.

    All pitched notes in each prefix measure must trace to preceding staffs.
    Prefix measures must be contiguous from 1; independent voices must agree.
    No reference pitch or source offset is searched. This is page-relative context.
    """
    by_call = {t["staff_call"]: t for t in transforms}
    vocal_calls = {
        r["staff_call"]
        for r in final
        if r["voice_index"] == voice and r.get("staff_call") is not None
    }
    if not vocal_calls:
        raise ValueError("Vocal voice has no retained staff provenance")

    def bounds(call):
        transform = by_call[call]
        ratio = transform["original_to_processed_ratio"][1]
        ys = [y * ratio for point in transform["staff_grid"] for y in point["y"]]
        return min(ys), max(ys)

    first_top = min(bounds(call)[0] for call in vocal_calls)
    preceding = {call for call in by_call if bounds(call)[1] < first_top}
    if not preceding:
        return {"measure_offset": 0, "leading_systems": 0, "prefix_votes": []}
    if any(
        any(len(point["y"]) != 10 for point in by_call[call]["staff_grid"]) for call in preceding
    ):
        raise ValueError("Leading context is not exclusively piano grand staffs")
    origin_calls = {}
    for row in final:
        key = (row["voice_index"], row["token_index"])
        origin_calls.setdefault(key, set()).add(row.get("staff_call"))
    exported_calls = {}
    for record in trace["records"]:
        origin = record["origin"]
        calls = origin_calls.get(tuple(origin), set()) if origin is not None else set()
        exported_calls[(record["part_index"], record["exported_index"])] = (
            next(iter(calls)) if len(calls) == 1 else None
        )
    votes = []
    seen_calls = set()
    for part_index, events in enumerate(exported_by_part):
        measures = {}
        for index, event in enumerate(events):
            measures.setdefault(event["physical_measure"], []).append(
                exported_calls.get((part_index, index))
            )
        prefix = sorted(
            measure
            for measure, calls in measures.items()
            if calls and all(call in preceding for call in calls)
        )
        if not prefix:
            continue
        if prefix != list(range(1, max(prefix) + 1)):
            raise ValueError("Leading measure evidence is not contiguous from page start")
        calls = {call for measure in prefix for call in measures[measure]}
        seen_calls.update(calls)
        votes.append(
            {"part_index": part_index, "measure_offset": len(prefix), "staff_calls": sorted(calls)}
        )
    if seen_calls != preceding or not votes or len({v["measure_offset"] for v in votes}) != 1:
        raise ValueError("Leading piano measure context is incomplete or inconsistent")
    return {
        "measure_offset": votes[0]["measure_offset"],
        "leading_systems": len(preceding),
        "prefix_votes": votes,
    }
