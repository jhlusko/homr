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
