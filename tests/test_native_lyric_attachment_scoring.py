from training.onnx.native_lyric_attachment_scoring import score_native_outputs


def event(measure=6, onset="1/2"):
    return {
        "physical_measure": measure,
        "onset_quarters": onset,
        "pitch": "C5",
        "alter": "0",
        "duration_quarters": "1/2",
    }


def test_page_global_verse_permutation_is_oracle_and_duplicates_are_capped():
    references = [
        {**event(), "verse": "1", "text": "oui"},
        {**event(), "verse": "2", "text": "yes"},
    ]
    predictions = [
        {"note": event(1), "physical_row_order": 1, "text": "yes"},
        {"note": event(1), "physical_row_order": 2, "text": "oui"},
    ]
    result = score_native_outputs(references, [*predictions, predictions[0]], 5)
    assert result["joint_literal_matches_after_oracle_page_verse_permutation"] == 2
    assert result["oracle_page_verse_mapping"] == {1: "2", 2: "1"}
    assert result["native_verse_id_accuracy"] is None
    assert result["detector_misses"] is None


def test_measure_onset_and_accidental_errors_are_not_realigned():
    references = [{**event(), "verse": "1", "text": "yes"}]
    for note in [event(2), event(1, "3/4"), {**event(1), "alter": "1"}]:
        result = score_native_outputs(
            references, [{"note": note, "physical_row_order": 1, "text": "yes"}], 5
        )
        assert result["joint_literal_matches_after_oracle_page_verse_permutation"] == 0


def test_oracle_mapping_cannot_change_between_systems_or_notes():
    references = [{**event(onset=onset), "verse": "1", "text": "yes"} for onset in ("0", "1")]
    predictions = [
        {"note": event(1, onset), "physical_row_order": row, "text": "yes"}
        for onset, row in [("0", 1), ("1", 2)]
    ]
    result = score_native_outputs(references, predictions, 5)
    assert result["joint_literal_matches_after_oracle_page_verse_permutation"] == 1
    assert result["joint_literal_matches_ignoring_verse"] == 2


def test_identical_timing_pitch_on_different_native_notes_is_not_certified():
    references = [
        {**event(), "source_note_ordinal": i, "verse": "1", "text": "yes"} for i in (1, 2)
    ]
    result = score_native_outputs(
        references, [{"note": event(1), "physical_row_order": 1, "text": "yes"}], 5
    )
    assert result["ambiguous_native_note_keys"] == 1
    assert result["joint_literal_matches_after_oracle_page_verse_permutation"] == 0
