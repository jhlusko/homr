import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.ocr.detector_masks import CLASS_INDEX
from training.ocr.prepare_ossq_real_train import allowed_scores, build_score, source_page


def test_earlier_and_frozen_holdouts_cannot_enter_training():
    audit = {"train_scores": ["sq1"], "heldout_scores": ["sq2"]}
    split = {"corpora": {"ossq_boxes": {"roles": {
        "selection": {"scores": ["sq3"]}, "test": {"scores": ["sq4"]}
    }}}}
    assert allowed_scores(audit, split) == {"sq1"}
    audit["train_scores"].append("sq3")
    with pytest.raises(ValueError, match="overlap"):
        allowed_scores(audit, split)


def test_ocr_direction_and_dynamic_map_to_five_class_mask(tmp_path: Path):
    scan_root = tmp_path / "scans"
    image = scan_root / "composer" / "sq1:0001.png"
    image.parent.mkdir(parents=True)
    pixels = np.full((20, 20), 255, dtype=np.uint8)
    pixels[2:8, 2:8] = 0
    assert cv2.imwrite(str(image), pixels)
    old_name = "/workspace/b0/ossq-omr/scores/composer/sq1:0001.png"
    doc = tmp_path / "sq1.json"
    doc.write_text(json.dumps({"score_id": "sq1", "matches": [
        {"kind": "tempo", "page_image": old_name,
         "box": {"left": 2, "top": 2, "width": 2, "height": 2}},
        {"kind": "dynamic", "page_image": old_name,
         "box": {"left": 5, "top": 5, "width": 2, "height": 2}},
    ]}))
    result = build_score((doc, scan_root, tmp_path / "masks"))
    assert result["pages"] == 1
    assert result["boxes"] == {"DirectionText": 1, "Dynamic": 1}
    mask = cv2.imread(str(tmp_path / "masks/sq1/sq1:0001.mask.png"), cv2.IMREAD_GRAYSCALE)
    assert mask[2, 2] == CLASS_INDEX["DirectionText"]
    assert mask[5, 5] == CLASS_INDEX["Dynamic"]
    assert mask[10, 10] == 0
    with pytest.raises(ValueError, match="does not belong"):
        source_page(old_name, scan_root, "sq2")


def test_cross_score_and_missing_preview_pages_are_counted_and_skipped(tmp_path: Path):
    doc = tmp_path / "sq1.json"
    doc.write_text(json.dumps({"score_id": "sq1", "matches": [
        {"kind": "dynamic", "page_image": "/workspace/b0/ossq-omr/scores/a/sq2:0001.png",
         "box": {"left": 1, "top": 1, "width": 2, "height": 2}},
        {"kind": "tempo", "page_image": "/workspace/b0/ossq-omr/scores/a/sq1:0001_teaser.png",
         "box": {"left": 1, "top": 1, "width": 2, "height": 2}},
    ]}))
    result = build_score((doc, tmp_path / "scans", tmp_path / "masks"))
    assert result["pages"] == 0
    assert result["skipped"] == {
        "cross_score_matches": 1, "missing_page_matches": 1, "missing_pages": 1
    }
