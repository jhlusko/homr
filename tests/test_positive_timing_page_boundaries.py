import copy

from training.onnx.prepare_positive_timing_pages import certify_leading_system


def rows():
    return [
        dict(
            status="ambiguous",
            boundary_certified=True,
            count_exact=True,
            observed_measures=5,
            detected_measures=5,
            scan_index=14,
            system_index=0,
            start_measure=64,
            end_measure=69,
        ),
        dict(
            status="aligned",
            boundary_certified=True,
            count_exact=True,
            observed_measures=6,
            detected_measures=6,
            scan_index=15,
            system_index=1,
            start_measure=69,
            end_measure=75,
        ),
    ]


def test_certified_count_resolves_leading_window_without_predictions():
    original = rows()
    result, receipt = certify_leading_system(original)
    assert result[0]["status"] == "aligned"
    assert (result[0]["start_measure"], result[0]["end_measure"]) == (64, 69)
    assert original[0]["status"] == "ambiguous"
    assert receipt["model_predictions_used"] is False


def test_uncertain_count_anchor_or_gap_cannot_certify_window():
    for index, key, value in [
        (0, "boundary_certified", False),
        (0, "count_exact", False),
        (0, "observed_measures", 4),
        (0, "start_measure", 63),
        (1, "start_measure", 70),
        (1, "status", "ambiguous"),
        (1, "scan_index", 16),
        (1, "system_index", 2),
    ]:
        original = rows()
        original[index][key] = value
        snapshot = copy.deepcopy(original)
        result, receipt = certify_leading_system(original)
        assert receipt is None
        assert result == snapshot
