import pytest

from homr.text_detector_classes import DIRECTION_CLASS_ORDER
from training.ocr.decide_direction_release import decide


def _row(matched, predicted, ground_truth):
    return {"matched": matched, "predicted": predicted, "ground_truth": ground_truth}


def _report(direction, lieder, dynamic, *, candidate=False):
    role = "test"
    return {
        "weights_sha256": "chosen", "class_order": DIRECTION_CLASS_ORDER,
        "split_manifest_sha256": "frozen",
        "corpora": {
            "ossq_boxes": {role: {"folded": {
                "DirectionText": _row(*direction), "Dynamic": _row(*dynamic)}}},
            "lieder_boxes": {role: {"folded": {"DirectionText": _row(*lieder)}}},
            "lieder_lyrics": {role: {"folded": {}}},
        },
    }


def test_frozen_gate_requires_recall_and_prediction_caps():
    parent = _report((83, 367, 421), (11, 87, 11), (465, 2533, 1660))
    e4 = _report((0, 33276, 421), (0, 3898, 11), (1101, 5728, 1660))
    chosen = {"selected": {"epoch": 1, "weights_sha256": "chosen"}}
    history = {"history": [{"valid": {
        "DirectionText": .5, "Dynamic": .5, "Lyrics": .5, "MeasureNumber": .5
    }}]}
    passing = _report((105, 367, 421), (11, 87, 11), (1018, 5728, 1660), candidate=True)
    result = decide(parent, e4, chosen, passing, history)
    assert result["gate_pass"] is True
    assert result["floors"] == {
        "ossq_direction_matches": 105,
        "lieder_direction_matches": 11,
        "ossq_dynamic_matches": 1018,
    }
    failing = _report((104, 367, 421), (11, 87, 11), (1018, 5728, 1660), candidate=True)
    assert decide(parent, e4, chosen, failing, history)["checks"]["ossq_direction_recall"] is False
    passing["corpora"]["ossq_boxes"]["selection"] = {}
    with pytest.raises(ValueError, match="test pages only"):
        decide(parent, e4, chosen, passing, history)


def test_test_read_must_apply_selection_thresholds_unchanged():
    parent = _report((83, 367, 421), (11, 87, 11), (465, 2533, 1660))
    e4 = _report((0, 33276, 421), (0, 3898, 11), (1101, 5728, 1660))
    history = {"history": [{"valid": {"DirectionText": .5, "Dynamic": .5, "Lyrics": .5}}]}
    thresholds = {"DirectionText": .71, "Dynamic": .64}
    chosen = {"selected": {"epoch": 1, "weights_sha256": "chosen", "thresholds": thresholds}}
    candidate = _report((105, 367, 421), (11, 87, 11), (1018, 5728, 1660), candidate=True)
    with pytest.raises(ValueError, match="calibrated thresholds"):
        decide(parent, e4, chosen, candidate, history)
    candidate["thresholds"] = dict(thresholds)
    assert decide(parent, e4, chosen, candidate, history)["thresholds"] == thresholds


def test_direction_only_mode_ignores_dynamic_entirely():
    # decision 13 (2026-09-25): this checkpoint ships DirectionText only; Dynamic comes
    # from e4 unchanged, so a Dynamic shortfall here must not fail the gate or even
    # appear in it - there is nothing new being tested for that class.
    parent = _report((83, 367, 421), (11, 87, 11), (465, 2533, 1660))
    e4 = _report((0, 33276, 421), (0, 3898, 11), (1101, 5728, 1660))
    chosen = {"selected": {"epoch": 1, "weights_sha256": "chosen"}}
    history = {"history": [{"valid": {
        "DirectionText": .5, "Dynamic": .5, "Lyrics": .5, "MeasureNumber": .5
    }}]}
    # Direction and Lieder clear the bar; Dynamic is far below every floor that would
    # apply in the combined-gate mode (0 matched, 0 ground truth even).
    candidate = _report((105, 367, 421), (11, 87, 11), (0, 0, 0), candidate=True)

    result = decide(parent, e4, chosen, candidate, history, direction_only=True)

    assert result["gate_pass"] is True
    assert "ossq_dynamic_matches" not in result["floors"]
    assert "ossq_dynamic_recall" not in result["checks"]
    assert "ossq_dynamic_count" not in result["checks"]
    assert result["direction_only"] is True


def test_direction_only_mode_still_requires_the_synthetic_dynamic_learning_floor():
    parent = _report((83, 367, 421), (11, 87, 11), (465, 2533, 1660))
    e4 = _report((0, 33276, 421), (0, 3898, 11), (1101, 5728, 1660))
    chosen = {"selected": {"epoch": 1, "weights_sha256": "chosen"}}
    broken_dynamic_history = {"history": [{"valid": {
        "DirectionText": .5, "Dynamic": 0, "Lyrics": .5, "MeasureNumber": .5
    }}]}
    candidate = _report((105, 367, 421), (11, 87, 11), (0, 0, 0), candidate=True)

    result = decide(parent, e4, chosen, candidate, broken_dynamic_history, direction_only=True)

    assert result["gate_pass"] is False
    assert result["checks"]["synthetic_learning_floor"] is False
