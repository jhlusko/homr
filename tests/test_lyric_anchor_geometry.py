import numpy as np
import pytest

from training.onnx.lyric_anchor_geometry import canvas_to_page, inverse_dewarp


def test_inverse_uses_same_triangles_as_forward_dewarp():
    source = [[0, 0], [10, 1], [0, 10], [10, 11]]
    destination = [[0, 0], [10, 0], [0, 10], [10, 10]]
    assert np.allclose(
        inverse_dewarp([5, 5], source, destination, [[0, 1, 2], [1, 3, 2]]), [5, 5.5]
    )
    with pytest.raises(ValueError, match="coverage"):
        inverse_dewarp([20, 5], source, destination, [[0, 1, 2], [1, 3, 2]])


def test_canvas_padding_crops_and_page_scaling_are_inverted():
    transform = dict(
        canvas=dict(offset=[0, 20], input_size=[100, 40], resized_size=[200, 80]),
        crop_origins=[[10, 20], [5, 6]],
        staff_scale=2,
        original_to_processed_ratio=[3, 3],
        dewarp=dict(source=None, destination=None, simplices=[]),
    )
    assert canvas_to_page([40, 40], transform) == [52.5, 54.0]
    with pytest.raises(ValueError, match="padding"):
        canvas_to_page([40, 10], transform)
    with pytest.raises(ValueError, match="finite"):
        canvas_to_page([float("nan"), 40], transform)
