import pytest

from training.onnx.lyric_staff_pitch_geometry import interpolate_notehead_positions


def test_whole_staff_mapping_uses_local_slope_and_preserves_stored_position():
    transform = {
        "staff_call": 0,
        "original_to_processed_ratio": [2, 3],
        "staff_grid": [{"x": 0, "y": [0, 10, 20, 30, 40]}, {"x": 100, "y": [10, 20, 30, 40, 50]}],
        "detected_notehead_positions": [{"center": [50, 30], "staff_position": 3}],
    }
    result = interpolate_notehead_positions(transform)[0]
    assert result["continuous_position"] == 4
    assert result["staff_position"] == 4
    assert transform["detected_notehead_positions"][0]["staff_position"] == 3


def test_grandstaff_is_rejected():
    transform = {
        "staff_call": 0,
        "original_to_processed_ratio": [1, 1],
        "staff_grid": [{"x": 0, "y": list(range(10))}],
        "detected_notehead_positions": [],
    }
    with pytest.raises(ValueError):
        interpolate_notehead_positions(transform)
