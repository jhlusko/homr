from training.onnx.link_lyric_anchor_sequences import exact_sequence_matches


def test_missing_notes_cannot_shift_source_ownership_by_index():
    reference = [dict(pitch="C4", duration_quarters="1"), dict(pitch="D4", duration_quarters="1")]
    assert not exact_sequence_matches(reference, [dict(pitch="D4", rhythm="note_4")])


def test_written_pitch_and_duration_must_both_match():
    reference = [dict(pitch="C4", duration_quarters="1")]
    assert exact_sequence_matches(reference, [dict(pitch="C4", rhythm="note_4")])
    assert not exact_sequence_matches(reference, [dict(pitch="D4", rhythm="note_4")])
    assert not exact_sequence_matches(reference, [dict(pitch="C4", rhythm="note_8")])
