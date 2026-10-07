from training.onnx.score_semantic_role_baseline import assess, summarize


def row(box, text="a tempo"):
    return dict(box=box, text=text, lyrics_rectangle_union_coverage=0.2)


def test_abstention_is_not_a_direction_prediction_and_ocr_miss_is_separate():
    reference = dict(box=[0, 0, 20, 10], reviewed_role="DirectionText")
    observed = row([[0, 0], [20, 0], [20, 10], [0, 10]])
    assert assess(reference, [], [observed])["stage"] == "gate-abstention"
    assert assess(reference, [], [observed])["prediction"] == "Unknown"
    assert assess(reference, [], [])["stage"] == "ocr-miss"
    selected = assess(reference, [observed], [])
    assert selected["prediction"] == "Lyrics"
    changed_role = {**reference, "reviewed_role": "Lyrics"}
    assert assess(changed_role, [observed], []) == selected


def test_reference_coverage_and_role_denominators_do_not_hide_misses():
    reference = dict(box=[0, 0, 20, 10])
    small = row([[0, 0], [9, 0], [9, 10], [0, 10]])
    assert assess(reference, [small], [])["stage"] == "ocr-miss"
    summary = summarize(
        [
            dict(reference_role="Lyrics", stage="lyric-candidate"),
            dict(reference_role="Lyrics", stage="ocr-miss"),
            dict(reference_role="DirectionText", stage="gate-abstention"),
        ]
    )
    assert summary["Lyrics"]["lyric_selection_rate"] == 0.5
    assert summary["DirectionText"]["lyric_candidates"] == 0
    assert summary["OtherText"]["lyric_selection_rate"] is None
