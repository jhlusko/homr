import xml.etree.ElementTree as ET

import pytest

from training.architecture.transformer.exact_timing_targets import HEADS
from training.onnx.build_ossq_pixel_review import measure_signature, resolve_symbolic_window
from training.onnx.qualify_ossq_training_timing import (
    combine_support,
    replay_payload,
    resolve_indexed_window,
    restore_timing_map,
)


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


def rest_part(duration=None, measure_rest=True, divisions="24"):
    part = ET.Element("part")
    measure = ET.SubElement(part, "measure")
    attrs = ET.SubElement(measure, "attributes")
    ET.SubElement(attrs, "divisions").text = divisions
    note = ET.SubElement(measure, "note")
    ET.SubElement(note, "rest", {"measure": "yes"} if measure_rest else {})
    if duration is not None:
        ET.SubElement(note, "duration").text = duration
    return part


def test_omitted_full_rest_uses_source_duration_without_mutating_inputs():
    source, segment = rest_part("96"), rest_part()
    before = ET.tostring(segment)
    mapping, restored = restore_timing_map(source, segment)
    assert mapping[None]["elapsed_quarters"] == "4"
    assert mapping[None]["full_measure_rest"] is True
    assert len(restored) == 1
    assert ET.tostring(segment) == before


def test_omitted_ordinary_rest_and_conflicting_explicit_duration_refused():
    with pytest.raises(ValueError, match="not a paired full-measure rest"):
        restore_timing_map(rest_part("96", False), rest_part(None, False))
    with pytest.raises(ValueError, match="timing differ"):
        restore_timing_map(rest_part("96"), rest_part("48"))


def test_two_different_missing_full_rest_durations_refused():
    source, segment = rest_part("96"), rest_part()
    source.append(rest_part("72").find("measure"))
    segment.append(rest_part().find("measure"))
    with pytest.raises(ValueError, match="Ambiguous"):
        restore_timing_map(source, segment)


def test_missing_sidecar_is_only_synthesized_for_rest_only_crop(tmp_path):
    path = tmp_path / "rests.txt"
    path.write_text("rest_1 _ _ _ _ upper\n")
    payload = replay_payload(path, True)
    assert payload["symbols"] == [dict(identity=["rest_1", "_", "_", "upper"], timing=None)]
    with pytest.raises(ValueError, match="Missing replay"):
        replay_payload(path, False)
    path.write_text("note_4 C4 _ _ _ upper\n")
    with pytest.raises(ValueError, match="non-rest"):
        replay_payload(path, True)


def test_missing_divisions_can_be_recovered_for_durationless_full_rest():
    source, segment = rest_part("96"), rest_part()
    segment.find("measure").remove(segment.find("measure/attributes"))
    mapping, restored = restore_timing_map(source, segment)
    assert mapping[None]["source_divisions"] == "24"
    assert len(restored) == 1


def test_source_grace_notes_remain_without_elapsed_timing():
    source, segment = rest_part(), rest_part()
    for part in (source, segment):
        note = part.find("measure/note")
        note.remove(note.find("rest"))
        ET.SubElement(note, "grace")
        ET.SubElement(note, "type").text = "16th"
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = "C"
        ET.SubElement(pitch, "octave").text = "4"
    mapping, restored = restore_timing_map(source, segment)
    assert mapping == {None: None}
    assert restored == []
