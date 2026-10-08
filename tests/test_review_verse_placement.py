from training.onnx.audit_review_verse_placement import exact_verse_matches


def test_literal_verse_scoring_does_not_silently_apply_oracle_mapping():
    note = {"physical_measure": 1, "onset_quarters": "0", "pitch": "C4", "duration_quarters": "1"}
    reference = {**note, "text": "hello", "verse": "2"}
    prediction = {"note": note, "text": "hello", "physical_row_order": 1}
    assert exact_verse_matches([reference], [prediction], 0) == 0
    prediction["physical_row_order"] = 2
    assert exact_verse_matches([reference], [prediction], 0) == 1
    assert exact_verse_matches([reference], [prediction, prediction], 0) == 1
