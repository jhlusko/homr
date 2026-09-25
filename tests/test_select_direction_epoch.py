import pytest

from training.ocr.select_direction_epoch import select


def _report(direction, dynamic, lieder, role="selection"):
    def row(matched, ground_truth):
        return {"matched": matched, "predicted": matched + 5, "ground_truth": ground_truth}

    return {
        "weights": "epoch.pth", "weights_sha256": "hash", "split_manifest_sha256": "split",
        "corpora": {
            "ossq_boxes": {role: {"folded": {
                "DirectionText": row(direction, 467), "Dynamic": row(dynamic, 1841)}}},
            "lieder_boxes": {role: {"folded": {"DirectionText": row(lieder, 11)}}},
        },
    }


def test_selection_uses_recall_floors_and_never_test():
    parent = _report(94, 367, 8)
    parent["corpora"]["ossq_boxes"]["test"] = {}
    parent["corpora"]["lieder_boxes"]["test"] = {}
    candidates = [_report(109, 331, 7), _report(110, 330, 8)]
    history = {"history": [{"valid": {"DirectionText": .5, "Dynamic": .5, "Lyrics": .5}}] * 2}
    decision = select(parent, candidates, history)
    assert decision["floors"] == {
        "ossq_direction": 109, "ossq_dynamic": 331, "lieder_direction": 7
    }
    assert decision["selected"]["epoch"] == 1
    assert decision["candidates"][1]["eligible"] is False
    with pytest.raises(ValueError, match="test access"):
        select(parent, [_report(110, 331, 8, role="test")], history)


def test_direction_only_mode_drops_the_dynamic_floor():
    # decision 13 (2026-09-25): Dynamic ships from e4 unchanged, so a candidate must not
    # be disqualified, or preferred, by a class it will never be used for in production.
    parent = _report(94, 367, 8)
    # candidate 2 has a WORSE Dynamic count than candidate 1 (330 < 331, below the old
    # floor) but a BETTER DirectionText count (110 > 109) - direction-only mode should
    # pick it anyway, where the non-direction-only rule (tested above) picks candidate 1.
    candidates = [_report(109, 331, 7), _report(110, 330, 8)]
    history = {"history": [{"valid": {"DirectionText": .5, "Dynamic": .5, "Lyrics": .5}}] * 2}

    decision = select(parent, candidates, history, direction_only=True)

    assert "ossq_dynamic" not in decision["floors"]
    assert decision["candidates"][1]["eligible"] is True
    assert decision["selected"]["epoch"] == 2
    assert "decision 13" in decision["criterion"]


def test_direction_only_mode_still_enforces_the_synthetic_learning_gate():
    # Dropping the real-page Dynamic floor is not the same as dropping the sanity check
    # that the Dynamic head did not break during training - that gate is unrelated to
    # whether production uses the head's output.
    parent = _report(94, 367, 8)
    candidates = [_report(110, 0, 8)]
    broken_dynamic_history = {"history": [{"valid": {"DirectionText": .5, "Dynamic": 0, "Lyrics": .5}}]}

    decision = select(parent, candidates, broken_dynamic_history, direction_only=True)

    assert decision["candidates"][0]["eligible"] is False
    assert decision["candidates"][0]["synthetic_gate_ok"] is False
    assert decision["selected"] is None
