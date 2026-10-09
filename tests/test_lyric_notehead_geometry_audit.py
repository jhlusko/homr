"""Ensure diagnostic geometry refuses competing or incomplete staves."""

import numpy as np
import pytest

from training.onnx.audit_lyric_notehead_geometry import staff_positions


def mask(lines):
    result = np.zeros((180, 100), dtype=np.uint8)
    result[lines, :] = 1
    return result


def test_single_staff_geometry():
    head = {"center": [50, 50]}
    result = staff_positions(mask([20, 30, 40, 50, 60]), [head])
    assert result[0]["staffPosition"] == 3


@pytest.mark.parametrize(
    "lines,y",
    [
        ([20, 30, 40, 50], 50),
        ([20, 30, 40, 50, 60, 100, 110, 120, 130, 140], 80),
        ([20, 30, 40, 50, 60], 52.4),
    ],
)
def test_incomplete_competing_or_off_grid_geometry_is_refused(lines, y):
    with pytest.raises(ValueError):
        staff_positions(mask(lines), [{"center": [50, y]}])


def test_clef_inherits_only_from_preceding_same_part():
    from training.onnx.audit_lyric_notehead_geometry import active_clef

    previous = {"index": 0, "partIndex": 0, "systemIndex": 0, "tokens": [["clef_G2"]]}
    current = {"index": 2, "partIndex": 0, "systemIndex": 1, "tokens": [["note_4", "G4"]]}
    other = {"index": 1, "partIndex": 1, "systemIndex": 0, "tokens": [["clef_F4"]]}
    assert active_clef([other, previous, current], current) == "clef_G2"
    with pytest.raises(ValueError):
        active_clef([other, current], current)


def test_midstaff_clef_change_is_refused():
    from training.onnx.audit_lyric_notehead_geometry import active_clef

    staff = {
        "index": 0,
        "partIndex": 0,
        "systemIndex": 0,
        "tokens": [["clef_G2"], ["note_4", "G4"], ["clef_F4"]],
    }
    with pytest.raises(ValueError):
        active_clef([staff], staff)
