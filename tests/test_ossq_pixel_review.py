import xml.etree.ElementTree as ET

import pytest

from training.onnx.build_ossq_pixel_review import (
    resolve_symbolic_window,
    resolve_window,
    resolve_with_neighbors,
)


def source(parts):
    root = ET.Element("score-partwise")
    for numbers in parts:
        part = ET.SubElement(root, "part")
        for number in numbers:
            ET.SubElement(part, "measure", number=str(number))
    return root


def test_source_window_uses_physical_indices_instead_of_measure_labels():
    assert resolve_window(source([[0, 7, 8, 9], [0, 7, 8, 9]]), [7, 8]) == (1, 3)


def test_ambiguous_or_inconsistent_source_windows_are_rejected():
    with pytest.raises(ValueError, match="unique and contiguous"):
        resolve_window(source([[1, 2, 1, 2]]), [1, 2])
    with pytest.raises(ValueError, match="unique and contiguous"):
        resolve_window(source([[1, 2, 3]]), [1, 3])
    with pytest.raises(ValueError, match="parts disagree"):
        resolve_window(source([[0, 1, 2], [1, 2]]), [1, 2])


def test_musical_anchor_disambiguates_restarted_measure_numbers_and_checks_accidentals():
    root = source([[1, 2, 1, 2], [1, 2, 1, 2]])
    for part in root.findall("part"):
        for index, measure in enumerate(part):
            note = ET.SubElement(measure, "note")
            ET.SubElement(note, "type").text = "quarter"
            pitch = ET.SubElement(note, "pitch")
            ET.SubElement(pitch, "step").text = "C"
            ET.SubElement(pitch, "octave").text = "4"
            ET.SubElement(pitch, "alter").text = "1" if index > 1 else "0"
    segment = ET.fromstring(ET.tostring(root))
    for part in segment.findall("part"):
        for measure in list(part)[:2]:
            part.remove(measure)
    assert resolve_symbolic_window(root, segment) == (2, 4)
    segment.find("part/measure/note/pitch/alter").text = "-1"
    with pytest.raises(ValueError, match="unique all-part"):
        resolve_symbolic_window(root, segment)


def test_neighbor_context_disambiguates_an_exact_repeated_passage(tmp_path):
    def musical(steps):
        root = source([list(range(len(steps))), list(range(len(steps)))])
        for part in root.findall("part"):
            for measure, step in zip(part, steps):
                note = ET.SubElement(measure, "note")
                ET.SubElement(note, "type").text = "quarter"
                pitch = ET.SubElement(note, "pitch")
                ET.SubElement(pitch, "step").text = step
                ET.SubElement(pitch, "octave").text = "4"
        return root

    root = musical(["C", "D", "C", "E"])
    for number, step in enumerate(["D", "C", "E"], 1):
        ET.ElementTree(musical([step])).write(tmp_path / f"sq1:0001:{number:04d}.musicxml")
    target = tmp_path / "sq1:0001:0002.musicxml"
    with pytest.raises(ValueError, match="unique all-part"):
        resolve_symbolic_window(root, ET.parse(target).getroot())
    start, end, evidence = resolve_with_neighbors(root, target)
    assert (start, end) == (2, 3)
    assert len(evidence) == 3
