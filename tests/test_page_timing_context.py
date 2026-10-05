import xml.etree.ElementTree as ET

import pytest

from training.onnx.audit_page_timing_context import edits, event_locations, meter_streams
from training.onnx.score_full_page_musicxml import compare


def test_timing_context_detects_rebarring_omitted_by_ordered_event_metric():
    note = {"measure": 0, "onset": "0", "duration": "1", "grace": False, "pitch": ("C", 4, "0")}
    later = dict(note, measure=1)
    assert compare([[note]], [[later]])["event"]["errors"] == 0
    assert edits([event_locations([note])], [event_locations([later])])["errors"] == 1
    shifted = dict(note, onset="1/3")
    assert edits([event_locations([note])], [event_locations([shifted])])["errors"] == 1


def test_meter_is_inherited_before_page_and_keeps_staff_specific_change():
    root = ET.fromstring(
        "<score-partwise><part><measure><attributes><staves>2</staves>"
        "<time><beats>6</beats><beat-type>4</beat-type></time></attributes></measure>"
        '<measure><attributes><time number="2"><beats>4</beats><beat-type>4</beat-type>'
        "</time></attributes></measure><measure/></part></score-partwise>"
    )
    streams = meter_streams(root, 1, 3)
    assert streams[0] == [(0, (("6",), ("4",), False)), (1, (("6",), ("4",), False))]
    assert streams[1] == [(0, (("4",), ("4",), False)), (1, (("4",), ("4",), False))]
    assert edits(streams, streams)["support"] == 4
    assert edits(streams, streams[:1])["errors"] == 2


def test_missing_meter_stays_unknown_and_extra_bar_counts_as_error():
    root = ET.fromstring("<score-partwise><part><measure/><measure/></part></score-partwise>")
    stream = meter_streams(root)
    assert stream == [[(0, None), (1, None)]]
    assert edits(stream, [stream[0] + [(2, None)]])["errors"] == 1


def test_mid_measure_meter_change_is_not_silently_relocated():
    root = ET.fromstring(
        "<score-partwise><part><measure><note><rest/></note><attributes>"
        "<time><beats>3</beats><beat-type>4</beat-type></time>"
        "</attributes></measure></part></score-partwise>"
    )
    with pytest.raises(ValueError, match="after measure material"):
        meter_streams(root)
