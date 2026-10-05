from training.onnx.experiment_cached_alias_xml import without_slurs


def test_non_slur_projection_ignores_phrase_alias_removal():
    import xml.etree.ElementTree as ET

    root = ET.fromstring(
        '<score-partwise><part><measure><note><pitch><step>C</step>'
        '<octave>4</octave></pitch><notations><slur type="start"/>'
        '<tied type="start"/></notations></note></measure></part>'
        '</score-partwise>'
    )
    before = without_slurs(root)
    root.find('.//notations').remove(root.find('.//notations/slur'))
    assert without_slurs(root) == before
