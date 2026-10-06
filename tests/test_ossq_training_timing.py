import xml.etree.ElementTree as ET

import pytest

from training.architecture.transformer.exact_timing_targets import HEADS
from training.onnx.build_ossq_pixel_review import measure_signature, resolve_symbolic_window
from training.onnx.qualify_ossq_training_timing import combine_support, resolve_indexed_window


def root(steps):
    score = ET.Element("score-partwise")
    for _ in range(2):
        part = ET.SubElement(score, "part")
        for step in steps:
            measure = ET.SubElement(part, "measure")
            note = ET.SubElement(measure, "note")
            ET.SubElement(note, "type").text = "quarter"
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = step
            ET.SubElement(pitch, "octave").text = "4"
    return score


def test_indexed_window_matches_original_resolver_and_refuses_repeats():
    source, segment = root("CDE"), root("DE")
    signatures = [
        [measure_signature(m) for m in p.findall("measure")] for p in source.findall("part")
    ]
    assert (
        resolve_indexed_window(signatures, segment)
        == resolve_symbolic_window(source, segment)
        == (1, 3)
    )
    source = root("CDCD")
    signatures = [
        [measure_signature(m) for m in p.findall("measure")] for p in source.findall("part")
    ]
    with pytest.raises(ValueError, match="unique"):
        resolve_indexed_window(signatures, root("CD"))


def test_combined_support_excludes_changed_and_validation_crops():
    lieder = {
        "groups": {
            "train/identical": {"heads": {h: {"a": 2} for h in HEADS}},
            "train/changed": {"heads": {h: {"b": 100} for h in HEADS}},
            "valid/identical": {"heads": {h: {"c": 100} for h in HEADS}},
        }
    }
    combined, unsupported = combine_support(
        lieder,
        {"heads": {h: {"a": 3, "d": 4} for h in HEADS}},
        {h: ["a", "b", "c", "d"] for h in HEADS},
    )
    for head in HEADS:
        assert combined[head] == {"a": 5, "d": 4}
        assert unsupported[head] == ["b", "c"]
