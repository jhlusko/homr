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


def test_wider_context_recovers_lines_obscured_by_short_fragment():
    image = np.zeros((100, 240), dtype=np.uint8)
    image[[20, 30, 40, 50, 60], :] = 1
    image[35, 100:140] = 1
    result = staff_positions(image, [{"center": [120, 50]}])
    assert result[0]["staffPosition"] == 3
    assert result[0]["supportingWindows"] == 2


def test_a_single_supported_window_is_insufficient():
    image = np.zeros((100, 240), dtype=np.uint8)
    image[[20, 30, 40, 50, 60], 108:133] = 1
    with pytest.raises(ValueError, match="Insufficient staff-window support"):
        staff_positions(image, [{"center": [120, 50]}])


def test_staff_windows_must_agree_on_diatonic_position(monkeypatch):
    from training.onnx import audit_lyric_notehead_geometry as audit

    def candidates(_mask, _x, _y, width):
        if width == 240:
            return []
        if width == 80:
            return [[23, 33, 43, 53, 63]]
        return [[20, 30, 40, 50, 60]]

    monkeypatch.setattr(audit, "local_staff_candidates", candidates)
    with pytest.raises(ValueError, match="disagrees across staff windows"):
        staff_positions(np.zeros((100, 240)), [{"center": [120, 50]}])


def test_long_range_lines_can_disambiguate_local_fragment(monkeypatch):
    from training.onnx import audit_lyric_notehead_geometry as audit

    def candidates(_mask, _x, _y, width):
        if width == 240:
            return [[20, 30, 40, 50, 60]]
        return [[20, 30, 40, 50, 60], [30, 40, 50, 60, 70]]

    monkeypatch.setattr(audit, "local_staff_candidates", candidates)
    assert staff_positions(np.zeros((100, 240)), [{"center": [120, 50]}])[0]["staffPosition"] == 3


def test_clipped_neighbouring_staff_head_does_not_poison_voice_geometry():
    image = np.zeros((160, 240), dtype=np.uint8)
    image[[60, 70, 80, 90, 100], :] = 1
    result = staff_positions(image, [{"center": [40, 5]}, {"center": [120, 90]}])
    assert len(result) == 1 and result[0]["center"] == [120, 90]
