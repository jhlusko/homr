import copy
import json
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from training.onnx.certify_page_extension import certify
from training.onnx.prepare_page_certification_extension import preview_window
from training.onnx.qualify_joint_export import sha


def test_preview_carries_staff_attributes_without_modifying_reference():
    root = ET.fromstring(
        '<score-partwise><part id="p"><measure number="1"><attributes>'
        "<divisions>12</divisions><staves>2</staves><key><fifths>-3</fifths></key>"
        "<time><beats>6</beats><beat-type>4</beat-type></time>"
        '<clef number="1"><sign>G</sign></clef><clef number="2"><sign>F</sign></clef>'
        '</attributes></measure><measure number="2"><print new-page="yes"/>'
        "<attributes><time><beats>4</beats><beat-type>4</beat-type></time>"
        '<clef number="1"><sign>C</sign></clef></attributes></measure>'
        '<measure number="3"/></part></score-partwise>'
    )
    before = ET.tostring(root)
    result = preview_window(root, 1, 3)
    assert ET.tostring(root) == before
    first = result.find("part/measure")
    assert first.findtext("attributes/divisions") == "12"
    assert len(first.findall(".//time")) == 1
    assert first.findtext(".//time/beats") == "4"
    assert {x.get("number"): x.findtext("sign") for x in first.findall(".//clef")} == {
        "1": "C",
        "2": "F",
    }
    assert first.find("print").get("new-page") is None
    with pytest.raises(ValueError, match="Invalid preview"):
        preview_window(root, 2, 4)


def fixture(tmp_path):
    source = tmp_path / "source.musicxml"
    ET.ElementTree(
        ET.fromstring('<score-partwise><part id="p"><measure/><measure/></part></score-partwise>')
    ).write(source)
    image = tmp_path / "page.png"
    Image.new("L", (30, 40)).save(image)
    evidence = tmp_path / "independent-alignment.json"
    evidence.write_text('{"model_predictions_used": false}')
    row = {
        "source": "lieder",
        "score": "IMSLP1",
        "image": str(image),
        "image_sha256": sha(image),
        "reference": str(source),
        "reference_sha256": sha(source),
        "reference_total_measures": 2,
        "proposed_start_measure": 0,
        "proposed_end_measure": 2,
        "systems": [
            {
                "system_index": 0,
                "start_measure": 0,
                "end_measure": 2,
                "status": "aligned",
                "count_exact": True,
                "boundary_certified": True,
            }
        ],
    }
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps([row]))
    (tmp_path / "selection-receipt.json").write_text(
        json.dumps({"selection_sha256": sha(selection), "inputs": {str(evidence): sha(evidence)}})
    )
    review = {
        "selection_sha256": sha(selection),
        "reviewer": "fixture",
        "model_predictions_used": False,
        "scores": {
            "IMSLP1": {
                "edition_identity_checked": True,
                "complete_page_checked": True,
                "part_order_checked": True,
                "image_sha256": sha(image),
                "reference_sha256": sha(source),
                "visible_system_measure_counts": [2],
                "boundary_evidence": ["checked fixture"],
                "review_artifacts": {},
            }
        },
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return selection, path, review, evidence


def test_certification_keeps_review_scope_and_rejects_changed_evidence(tmp_path):
    selection, review_path, _, evidence = fixture(tmp_path)
    certify(selection, review_path, tmp_path / "out")
    pages = json.loads((tmp_path / "out/pages.json").read_text())
    assert pages[0]["end_measure"] == 2
    assert "not exhaustive" in pages[0]["certification_scope"]
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        certify(selection, review_path, tmp_path / "other")


def test_certification_rejects_missing_review_or_prediction_assisted_review(tmp_path):
    selection, review_path, review, _ = fixture(tmp_path)
    missing = copy.deepcopy(review)
    missing["scores"]["IMSLP1"]["part_order_checked"] = False
    review_path.write_text(json.dumps(missing))
    with pytest.raises(ValueError, match="Missing visual review"):
        certify(selection, review_path, tmp_path / "out")
    review["model_predictions_used"] = True
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Independent reviewer"):
        certify(selection, review_path, tmp_path / "out")


def test_certification_rejects_incorrect_visible_system_count(tmp_path):
    selection, review_path, review, _ = fixture(tmp_path)
    review["scores"]["IMSLP1"]["visible_system_measure_counts"] = [1]
    review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Visual system counts"):
        certify(selection, review_path, tmp_path / "out")
