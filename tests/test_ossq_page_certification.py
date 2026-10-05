import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from training.onnx.certify_ossq_opening_pages import certify, source_opening_windows
from training.onnx.qualify_joint_export import sha
from training.onnx.score_paired_page_diagnostic import read_reference, score_pair


def quartet():
    root = ET.Element("score-partwise")
    for index in range(4):
        part = ET.SubElement(root, "part", id=str(index))
        for number in range(1, 6):
            measure = ET.SubElement(part, "measure", number=str(number))
            if number in (3, 5):
                ET.SubElement(
                    measure, "print", **{"new-system" if number == 3 else "new-page": "yes"}
                )
    return root


def test_all_parts_must_agree_on_complete_opening_page():
    root = quartet()
    assert source_opening_windows(root) == ([(0, 2), (2, 4)], 5)
    root.findall("part")[2].findall("measure")[2].find("print").set("new-system", "no")
    with pytest.raises(ValueError, match="boundaries disagree"):
        source_opening_windows(root)


def test_source_numbering_and_generated_references_are_rejected():
    root = quartet()
    root.findall("part")[1].findall("measure")[1].set("number", "99")
    with pytest.raises(ValueError, match="identities disagree"):
        source_opening_windows(root)
    root = quartet()
    ET.SubElement(root, "software").text = "HOMR"
    with pytest.raises(ValueError, match="cannot be reference"):
        source_opening_windows(root)


def test_certification_rejects_wrong_counts_and_changed_reviewed_pixels(tmp_path):
    source, image = tmp_path / "source.musicxml", tmp_path / "page.png"
    ET.ElementTree(quartet()).write(source)
    Image.new("L", (30, 40)).save(image)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            [
                {
                    "score": "sq1",
                    "reference": str(source),
                    "image": str(image),
                    "dimensions": [30, 40],
                    "pixel_budget": 2000,
                    "proposed_start_measure": 0,
                    "proposed_end_measure": 4,
                }
            ]
        )
    )
    checked = {
        "edition_identity_checked": True,
        "complete_page_checked": True,
        "part_order_checked": True,
        "reference_sha256": sha(source),
        "image_sha256": sha(image),
        "visible_system_measure_counts": [1, 3],
    }
    review = tmp_path / "review.json"

    def save():
        review.write_text(
            json.dumps(
                {
                    "selection_sha256": sha(selection),
                    "reviewer": "fixture",
                    "model_predictions_used": False,
                    "scores": {"sq1": checked},
                }
            )
        )

    save()
    with pytest.raises(ValueError, match="counts disagree"):
        certify(selection, review, tmp_path / "out")
    checked["visible_system_measure_counts"] = [2, 2]
    save()
    certify(selection, review, tmp_path / "out")
    Image.new("L", (30, 40), 255).save(image)
    with pytest.raises(ValueError, match="changed"):
        certify(selection, review, tmp_path / "other")


def test_plain_xml_reference_rejects_homr_output(tmp_path):
    source = tmp_path / "source.musicxml"
    ET.ElementTree(quartet()).write(source)
    assert read_reference(source).tag == "score-partwise"
    root = quartet()
    ET.SubElement(root, "software").text = "homr"
    ET.ElementTree(root).write(source)
    with pytest.raises(ValueError, match="recognizer-generated"):
        read_reference(source)


def test_failed_decode_has_no_zero_error_metrics_and_protocol_mismatch_fails(tmp_path):
    source = tmp_path / "source.musicxml"
    ET.ElementTree(quartet()).write(source)
    page = {
        "reference": str(source),
        "reference_sha256": sha(source),
        "start_measure": 0,
        "end_measure": 4,
    }
    for arm in ("accepted", "candidate"):
        folder = tmp_path / arm
        folder.mkdir()
        (folder / "identity.json").write_text(
            json.dumps(
                {
                    "page": page,
                    "arm": arm,
                    "policy": {},
                    "code_sha256": {},
                    "segmentation": {},
                    "threads": 1,
                    "device": "CPU",
                }
            )
        )
        (folder / "result.json").write_text(
            json.dumps({"status": "failed", "seconds": 2, "error": "resize failure"})
        )
    result = score_pair(page, tmp_path)
    assert result["successful_arms"] == 0 and result["attempted_arms"] == 2
    assert all("metrics" not in arm for arm in result["arms"].values())
    path = tmp_path / "candidate/identity.json"
    identity = json.loads(path.read_text())
    identity["threads"] = 2
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="protocol"):
        score_pair(page, tmp_path)


def test_pair_runner_preserves_failures_and_resumes_without_redecoding(tmp_path, monkeypatch):
    from training.onnx import run_paired_page_diagnostics as runner

    source = tmp_path / "source.musicxml"
    ET.ElementTree(quartet()).write(source)
    image = tmp_path / "page.png"
    Image.new("L", (10, 10)).save(image)
    page = {
        "score": "sq1",
        "reference": str(source),
        "reference_sha256": sha(source),
        "image": str(image),
        "image_sha256": sha(image),
        "start_measure": 0,
        "end_measure": 4,
    }
    pages, models = tmp_path / "pages.json", tmp_path / "models.json"
    pages.write_text(json.dumps([page]))
    models.write_text(json.dumps({"accepted": {}, "candidate": {}}))
    calls = []

    class FailedWorker:
        def __init__(self, command, **kwargs):
            calls.append(command)
            folder = Path(command[command.index("--out") + 1])
            arm = command[command.index("--arm") + 1]
            (folder / "identity.json").write_text(
                json.dumps(
                    {
                        "page": page,
                        "arm": arm,
                        "models": {},
                        "policy": {},
                        "code_sha256": {},
                        "segmentation": {},
                        "threads": 1,
                        "device": "CPU",
                    }
                )
            )
            (folder / "result.json").write_text(
                json.dumps({"status": "failed", "seconds": 2, "error": "fixture failure"})
            )

        def poll(self):
            return 1

    monkeypatch.setattr(runner.subprocess, "Popen", FailedWorker)
    output = tmp_path / "run"
    runner.run(pages, models, output)
    summary = json.loads((output / "summary.json").read_text())
    assert len(calls) == 2 and summary["attempted_arms"] == 2
    assert summary["successful_arms"] == 0 and summary["paired_successful_pages"] == 0
    runner.run(pages, models, output)
    assert len(calls) == 2
    pages.write_text(json.dumps([]))
    with pytest.raises(ValueError, match="cannot resume"):
        runner.run(pages, models, output)
