from training.ocr.calibrate_direction_thresholds import (
    choose_thresholds,
    reference_caps,
    threshold_for_cap,
)
from training.ocr.detector_inference import PredictedBox
from training.ocr.eval_folded_directions import apply_thresholds


def _box(label, confidence):
    return PredictedBox(label, 0, 0, 10, 10, confidence)


def test_threshold_keeps_at_most_cap_boxes_and_drops_ties_at_the_cut():
    confidences = [.9, .8, .7, .7, .6]
    assert threshold_for_cap(confidences, 5) == 0.0
    kept = [c for c in confidences if c >= threshold_for_cap(confidences, 2)]
    assert kept == [.9, .8]
    # a tie straddling the cap is dropped whole rather than overshooting it
    kept = [c for c in confidences if c >= threshold_for_cap(confidences, 3)]
    assert kept == [.9, .8]


def test_stricter_corpus_sets_the_shared_class_threshold():
    ossq = [({}, [], [_box("DirectionText", c) for c in (.9, .8, .7, .6)])]
    lieder = [({}, [], [_box("DirectionText", c) for c in (.95, .85, .5)])]
    caps = {"DirectionText": {"ossq_boxes": 3, "lieder_boxes": 1}}
    thresholds = choose_thresholds({"ossq_boxes": ossq, "lieder_boxes": lieder}, caps)
    assert .85 < thresholds["DirectionText"] <= .95
    kept = apply_thresholds(ossq[0][2] + lieder[0][2], thresholds)
    assert [box.confidence for box in kept] == [.9, .95]


def test_seven_class_predictions_fold_before_thresholding():
    boxes = [_box("Tempo", .4), _box("StaffText", .8), _box("Dynamic", .4)]
    kept = apply_thresholds(boxes, {"DirectionText": .5})
    assert [box.label for box in kept] == ["StaffText", "Dynamic"]


def test_caps_come_from_parent_direction_and_e4_dynamic_on_selection():
    def report(direction, lieder, dynamic):
        row = lambda predicted: {"predicted": predicted}
        return {"corpora": {
            "ossq_boxes": {"selection": {"folded": {
                "DirectionText": row(direction), "Dynamic": row(dynamic)}}},
            "lieder_boxes": {"selection": {"folded": {"DirectionText": row(lieder)}}},
        }}
    caps = reference_caps(report(512, 38, 3262), report(9999, 999, 10023))
    assert caps == {
        "DirectionText": {"ossq_boxes": 512, "lieder_boxes": 38},
        "Dynamic": {"ossq_boxes": 10023},
    }
