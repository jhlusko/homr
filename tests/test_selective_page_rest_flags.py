import xml.etree.ElementTree as ET

from training.onnx.audit_selective_page_timing import compare_flags, rest_flags


def test_rest_flags_use_physical_staff_and_native_onset_duration():
    root = ET.fromstring("""<score-partwise><part><measure>
    <attributes><divisions>2</divisions><staves>2</staves></attributes>
    <note><rest measure="yes"/><duration>8</duration><staff>1</staff></note>
    <backup><duration>8</duration></backup>
    <note><rest/><duration>4</duration><staff>2</staff></note>
    <note><rest/><duration>4</duration><staff>2</staff></note>
    </measure></part></score-partwise>""")
    values = rest_flags(root)
    assert values == {
        (1, 0, "0", "4"): [True],
        (2, 0, "0", "2"): [False],
        (2, 0, "2", "2"): [False],
    }
    candidate = {**values, (1, 0, "0", "4"): [True]}
    baseline = {**values, (1, 0, "0", "4"): [False]}
    report = compare_flags(values, baseline, candidate)
    assert report["improvements"] == 1
    assert report["regressions"] == 0
    assert report["uniquely_matched"] == 3
    candidate[(1, 0, "0", "4")] = [True, True]
    report = compare_flags(values, baseline, candidate)
    assert report["unmatched"] == 1
    assert report["reference_support"] == 3
