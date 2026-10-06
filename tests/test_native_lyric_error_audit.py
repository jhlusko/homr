from collections import Counter

from training.onnx.audit_native_lyric_errors import decompose


def event(text="word", **changes):
    return {
        "physical_measure": 1,
        "onset_quarters": "0",
        "pitch": "C4",
        "duration_quarters": "1",
        "alter": "0",
        "source_note_ordinal": 1,
        "verse": "1",
        "text": text,
        **changes,
    }


def prediction(text, note=None):
    return {"physical_row_order": 1, "text": text, "note": note or event()}


def test_exact_matches_reserved_before_text_capacity_and_duplicates_capped():
    refs = [event("wrong"), event("right"), event("other")]
    records = decompose(refs, [prediction("right"), prediction("typo")], [event()], 0, {"1": "1"})
    assert Counter(r["category"] for r in records) == {
        "exact": 1,
        "text_mismatch_at_exact_note_and_oracle_verse": 1,
        "no_attachment_at_exact_decoded_identity": 1,
    }
    assert records[1]["category"] == "exact"


def test_missing_attachment_is_separate_from_missing_decoded_identity():
    refs = [event(), event(onset_quarters="1", source_note_ordinal=2)]
    records = decompose(refs, [], [event()], 0, {})
    assert [r["category"] for r in records] == [
        "no_attachment_at_exact_decoded_identity",
        "native_identity_absent_from_decoded_voice",
    ]


def test_ambiguous_native_notes_remain_unscored_and_offsets_are_fixed():
    refs = [event(physical_measure=3), event(physical_measure=3, source_note_ordinal=2)]
    records = decompose(refs, [prediction("word")], [event()], 2, {"1": "1"})
    assert all(r["category"] == "ambiguous_native_identity" for r in records)
    records = decompose([event(physical_measure=3)], [prediction("word")], [event()], 1, {"1": "1"})
    assert records[0]["category"] == "native_identity_absent_from_decoded_voice"


def test_broad_normalization_is_only_a_diagnostic_and_preserves_literal_error():
    records = decompose([event("we're")], [prediction("were")], [event()], 0, {"1": "1"})
    assert records[0]["category"] == "text_mismatch_at_exact_note_and_oracle_verse"
    assert records[0]["normalized_text_equal"]
    assert not records[0]["nfc_trimmed_text_equal"]
