import xml.etree.ElementTree as ET

from training.onnx.audit_page_notation_policy import anchors, compare_common


def note(stem="up", pitch="C", slur="above"):
    return f'<note><pitch><step>{pitch}</step><octave>4</octave></pitch><duration>1</duration><stem>{stem}</stem><notations><slur type="start" placement="{slur}"/></notations></note>'


def root(notes):
    return ET.fromstring(
        "<score-partwise><part><measure><attributes><divisions>1</divisions></attributes>"
        + notes
        + "</measure></part></score-partwise>"
    )


def test_matching_is_fixed_by_pitch_and_timing_not_notation():
    reference = anchors(root(note()))
    changed = anchors(root(note("down", slur="below")))
    result = compare_common(
        reference, {"changed": changed, "pitch_error": anchors(root(note(pitch="D")))}
    )
    assert result["unique_common_pitched_anchors"] == 0
    result = compare_common(reference, {"changed": changed})
    assert result["scores"]["changed"] == {
        "stem": {"errors": 1, "support": 1},
        "slur_start_side": {
            "errors": 1,
            "support": 1,
            "missing_or_extra_start": 0,
            "unknown_single_side": 0,
            "wrong_single_side": 1,
        },
    }


def test_duplicate_unisons_are_not_matched_by_stem_or_slur_side():
    xml = root(note() + "<backup><duration>1</duration></backup>" + note("down"))
    rows = anchors(xml)
    assert compare_common(rows, {"same": rows})["unique_common_pitched_anchors"] == 0
    assert compare_common(rows, {"same": rows})["reference_ambiguous_anchor_keys"] == 1


def test_missing_explicit_reference_notation_is_not_inferred():
    xml = root(note())
    xml.find(".//note").remove(xml.find(".//stem"))
    xml.find(".//slur").attrib.pop("placement")
    result = compare_common(anchors(xml), {"pred": anchors(root(note()))})
    assert result["scores"]["pred"]["stem"]["support"] == 0
    assert result["scores"]["pred"]["slur_start_side"]["support"] == 0
