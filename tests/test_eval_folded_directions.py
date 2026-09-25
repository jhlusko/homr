from training.ocr.eval_folded_directions import reference_label
from homr.text_detector_classes import DIRECTION_CLASS_ORDER, LEGACY_CLASS_ORDER


def test_reference_labels_follow_checkpoint_scheme():
    for label in ("Tempo", "StaffText", "Expression", "SystemText"):
        assert reference_label(label, DIRECTION_CLASS_ORDER) == "DirectionText"
    assert reference_label("SystemText", LEGACY_CLASS_ORDER) == "StaffText"
    assert reference_label("Tempo", LEGACY_CLASS_ORDER) == "Tempo"
    assert reference_label("Expression", LEGACY_CLASS_ORDER) == "Expression"
    assert reference_label("Dynamic", DIRECTION_CLASS_ORDER) == "Dynamic"
