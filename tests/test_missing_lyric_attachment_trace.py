from training.onnx.trace_missing_lyric_attachments import trace_target


def fixture():
    event = {
        "physical_measure": 1,
        "onset_quarters": "0",
        "pitch": "C4",
        "alter": "0",
        "duration_quarters": "1",
        "verse": "1",
    }
    final = [
        {
            "voice_index": 0,
            "token_index": 3,
            "staff_call": 0,
            "page_coordinate": [10, 20],
            "system_index": 0,
        }
    ]
    prediction = {
        "export_trace": {"records": [{"part_index": 0, "exported_index": 0, "origin": [0, 3]}]},
        "systems": [{"staff_call": 0, "head_to_final_note": {"0": 0}, "head_xs": [10]}],
        "attachment_trace": [],
    }
    return event, final, prediction


def test_trace_separates_association_abstention_from_missing_syllable():
    event, final, prediction = fixture()
    assert (
        trace_target(event, [event], final, prediction, 0, {"1": "1"})["stage"]
        == "no_routed_syllable_selects_accepted_head"
    )
    prediction["systems"][0]["head_to_final_note"] = {}
    assert (
        trace_target(event, [event], final, prediction, 0, {"1": "1"})["stage"]
        == "no_accepted_notehead_association"
    )


def test_other_row_candidate_is_not_a_native_verse_match():
    event, final, prediction = fixture()
    prediction["attachment_trace"] = [
        {"staff_call": 0, "nearest_head_index": 0, "physical_row_order": 2, "text": "word"}
    ]
    assert (
        trace_target(event, [event], final, prediction, 0, {"1": "1", "2": "2"})["stage"]
        == "nearest_head_candidates_only_in_other_physical_rows"
    )
    prediction["attachment_trace"][0]["physical_row_order"] = 1
    assert (
        trace_target(event, [event], final, prediction, 0, {"1": "1"})["stage"]
        == "candidate_at_expected_row_requires_capacity_audit"
    )


def test_nonunique_export_identity_fails_closed():
    event, final, prediction = fixture()
    assert (
        trace_target(event, [event, event], final, prediction, 0, {})["stage"]
        == "nonunique_exported_identity"
    )
    prediction["export_trace"]["records"][0]["origin"] = None
    assert (
        trace_target(event, [event], final, prediction, 0, {})["stage"] == "nonunique_export_origin"
    )
