import pytest

from training.onnx.lyric_vocal_context import leading_measure_context, select_vocal_voice


def transform(call, top, lines=5):
    return {
        "staff_call": call,
        "original_to_processed_ratio": [1, 1],
        "staff_grid": [{"x": x, "y": [top + i * 10 for i in range(lines)]} for x in [0, 100]],
    }


def fixture():
    transforms = [transform(0, 0, 10), transform(1, 300, 10), transform(2, 150)]
    final = [
        {"voice_index": 0, "token_index": 0, "staff_call": 0},
        {"voice_index": 0, "token_index": 1, "staff_call": 0},
        {"voice_index": 0, "token_index": 2, "staff_call": 1},
        {"voice_index": 1, "token_index": 0, "staff_call": 2},
    ]
    trace = {
        "records": (
            [{"part_index": 0, "exported_index": i, "origin": [0, i]} for i in range(3)]
            + [{"part_index": 1, "exported_index": 0, "origin": [1, 0]}]
        )
    }
    events = [[{"physical_measure": i} for i in [1, 2, 3]], [{"physical_measure": 1}]]
    return transforms, final, trace, events


def test_vocal_voice_can_follow_a_piano_voice_without_native_labels():
    transforms, final, _, _ = fixture()
    selected = select_vocal_voice(
        final, transforms, {"routed": [{"staff_call": 2, "physical_row_order": 1}]}
    )
    assert selected["voice_index"] == 1
    assert not selected["native_input_used"]
    assert not selected["semantic_lyric_role_certified"]


def test_ambiguous_text_supported_single_staff_voices_fail_closed():
    transforms, final, _, _ = fixture()
    transforms[0] = transform(0, 0)
    transforms[1] = transform(1, 300)
    with pytest.raises(ValueError, match="unique"):
        select_vocal_voice(
            final,
            transforms,
            {"routed": [{"staff_call": c, "physical_row_order": 1} for c in [0, 2]]},
        )


def test_prefix_is_derived_from_traced_preceding_piano_measures():
    transforms, final, trace, events = fixture()
    result = leading_measure_context(1, final, transforms, trace, events)
    assert result["measure_offset"] == 2
    assert result["leading_systems"] == 1
    assert leading_measure_context(0, final, transforms, trace, events)["measure_offset"] == 0


def test_missing_provenance_and_gapped_prefix_are_not_filled_from_references():
    transforms, final, trace, events = fixture()
    final[0].pop("staff_call")
    with pytest.raises(ValueError, match="contiguous"):
        leading_measure_context(1, final, transforms, trace, events)
    transforms, final, trace, events = fixture()
    events[0][0]["physical_measure"] = 4
    with pytest.raises(ValueError, match="contiguous"):
        leading_measure_context(1, final, transforms, trace, events)


def test_preceding_non_piano_staff_is_not_an_inferred_introduction():
    transforms, final, trace, events = fixture()
    transforms[0] = transform(0, 0)
    with pytest.raises(ValueError, match="grand staffs"):
        leading_measure_context(1, final, transforms, trace, events)


def test_recurring_support_can_outweigh_one_isolated_piano_text_region():
    transforms = [transform(i, i * 150) for i in range(4)]
    final = [{"voice_index": i // 2, "token_index": i, "staff_call": i} for i in range(4)]
    routing = {"routed": [{"staff_call": i, "physical_row_order": 1} for i in [0, 2, 3]]}
    result = select_vocal_voice(final, transforms, routing)
    assert result["voice_index"] == 1
    assert result["selection_basis"] == "unique_recurring_text_support_across_prepared_staffs"
    routing["routed"].append({"staff_call": 1, "physical_row_order": 1})
    with pytest.raises(ValueError, match="unique"):
        select_vocal_voice(final, transforms, routing)
