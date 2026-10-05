import pytest

from training.onnx.score_lyric_rows import normalize, row_for_box


def test_layout_normalization_keeps_accents_apostrophes_and_punctuation():
    assert normalize(" Si j’é - tais _ ") == "sij'étais"
    assert normalize("fill · it") == "fillit"
    assert normalize("weih’n,") == "weih'n,"
    assert normalize("né") != normalize("ne")


def test_spatial_assignment_does_not_use_predicted_text():
    rows = [{"id": "upper", "box": [0, 0, 20, 10]}, {"id": "lower", "box": [0, 11, 20, 20]}]
    assert row_for_box([[1, 2], [10, 2], [10, 6], [1, 6]], rows) == "upper"
    assert row_for_box([[1, 25], [10, 25], [10, 30], [1, 30]], rows) is None
    with pytest.raises(ValueError, match="ambiguous"):
        row_for_box([[1, 2], [10, 2], [10, 6], [1, 6]], rows + rows[:1])
