import pytest

from training.onnx.lyric_complete_staff_context import complete_routing_transforms


def transform(call, ys):
    return {
        "staff_call": call,
        "original_to_processed_ratio": [1, 1],
        "staff_grid": [{"x": x, "y": ys} for x in [0, 100]],
    }


def test_missing_physical_staff_is_a_boundary_not_a_decoded_voice():
    upper, lower, vocal, absent = [list(range(top, top + 41, 10)) for top in [0, 80, 160, 240]]
    runtime = [transform(0, upper + lower), transform(1, vocal)]
    raw = [transform(i, ys) for i, ys in enumerate([upper, lower, vocal, absent])]
    complete, evidence = complete_routing_transforms(runtime, raw)
    assert complete[:2] == runtime
    assert evidence["added_boundaries"] == 1
    assert complete[-1]["staff_call"] < 0
    assert complete[-1]["routing_boundary_only"]
    assert not complete[-1]["decoded_note_ownership"]


def test_unreproduced_runtime_staff_or_duplicate_correspondence_fails_closed():
    staff = transform(0, [0, 10, 20, 30, 40])
    with pytest.raises(ValueError, match="reproduce"):
        complete_routing_transforms([staff], [transform(1, [3, 13, 23, 33, 43])])
    with pytest.raises(ValueError, match="ambiguous"):
        complete_routing_transforms([staff], [staff, staff])
