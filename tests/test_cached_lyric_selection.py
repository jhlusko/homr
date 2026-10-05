import pytest

from training.onnx.audit_cached_slur_placement import classify
from training.onnx.gate_cached_lyric_lines import coverage


def test_overlap_uses_union_without_double_counting():
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    region = {"left": 0, "top": 0, "right": 8, "bottom": 10}
    assert coverage(box, [region, region]) == pytest.approx(0.8)
    other = {"left": 8, "top": 0, "right": 10, "bottom": 10}
    assert coverage(box, [region, other]) == pytest.approx(1)
    assert coverage(box, []) == 0


def test_degenerate_region_is_rejected():
    with pytest.raises(ValueError, match="Empty"):
        coverage([[0, 0], [0, 10]], [])


def test_side_is_not_borrowed_from_non_start_event():
    payload = {"final_notation_slurs": [["none", "above"], ["stop", "below"]]}
    assert classify(payload) == "no_matching_structured_start_slot"
    payload = {"final_notation_slurs": [["start", "unspecified"]]}
    assert classify(payload) == "matching_start_has_unspecified_side"
