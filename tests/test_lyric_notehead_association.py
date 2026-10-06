import pytest

from training.onnx.lyric_notehead_association import associate_x


def match(a, b):
    return associate_x(a, b, max_distance=20, ambiguity_margin=5)


def test_missing_head_does_not_shift_later_notes():
    result = match([0, 100, 200], [2, 198])
    assert result["pairs"] == [(0, 0), (2, 1)]
    assert result["unresolved_decoded"] == [1]


def test_extra_head_and_missing_decode_preserve_identity():
    assert match([0, 200], [2, 100, 198])["pairs"] == [(0, 0), (1, 2)]


def test_chords_and_near_ties_are_unresolved():
    assert match([100, 100], [100, 100])["pairs"] == []
    assert match([100], [98, 102])["pairs"] == []


def test_large_error_and_invalid_geometry():
    assert match([100], [130])["pairs"] == []
    with pytest.raises(ValueError):
        match([float("nan")], [1])
