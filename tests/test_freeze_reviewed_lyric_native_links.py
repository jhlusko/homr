import json

import pytest

from training.onnx.freeze_reviewed_lyric_native_links import freeze, sha


def fixture(tmp_path):
    review = tmp_path / "review"
    review.mkdir()
    source, scan = review / "native.xml", review / "scan.png"
    source.write_text("native source")
    scan.write_bytes(b"frozen scan")
    contacts = review / "inspection"
    contacts.mkdir()
    contact = contacts / "score-system1.png"
    contact.write_bytes(b"reviewed pixels")
    report = {
        "model_predictions_used": False,
        "pages": [
            {
                "score": "score",
                "image": str(scan),
                "image_sha256": sha(scan),
                "native_source": str(source),
                "native_source_sha256": sha(source),
                "start_measure": 0,
                "end_measure": 2,
                "page_native_lyric_targets": 3,
                "systems": [{"index": 0, "start_measure": 0, "end_measure": 2}],
            }
        ],
    }
    (review / "report.json").write_text(json.dumps(report))
    approval = {"wholeSetApproved": True, "reportSha256": sha(review / "report.json")}
    (review / "owner-review-whole-set.json").write_text(json.dumps(approval))
    (review / "manual-system-context.json").write_text("{}")
    inspection = {
        "reportSha256": sha(review / "report.json"),
        "ownerApprovalSha256": sha(review / "owner-review-whole-set.json"),
        "manualContextSha256": sha(review / "manual-system-context.json"),
        "systems": [
            {
                "score": "score",
                "system": 1,
                "nativeStart": 1,
                "nativeEnd": 2,
                "status": "passage_and_boundary_visual_checked",
                "contactSha256": sha(contact),
            }
        ],
        "textExceptions": [],
    }
    (review / "visual-inspection-complete.json").write_text(json.dumps(inspection))
    return review


def test_freeze_keeps_scope_and_refuses_overwrite(tmp_path):
    review = fixture(tmp_path)
    receipt = freeze(review, tmp_path / "out")
    assert receipt["nativeTargets"] == 3
    assert receipt["release_gate_passed"] is False
    assert receipt["pixel_link_certified"] is False
    with pytest.raises(FileExistsError):
        freeze(review, tmp_path / "out")


@pytest.mark.parametrize("file", ["native.xml", "scan.png", "inspection/score-system1.png"])
def test_changed_source_or_reviewed_pixels_refused(tmp_path, file):
    review = fixture(tmp_path)
    (review / file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        freeze(review, tmp_path / "out")


def test_partial_approval_is_not_whole_set(tmp_path):
    review = fixture(tmp_path)
    p = review / "owner-review-whole-set.json"
    doc = json.loads(p.read_text())
    doc["wholeSetApproved"] = False
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Whole-set"):
        freeze(review, tmp_path / "out")
