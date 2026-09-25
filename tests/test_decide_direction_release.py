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
