import xml.etree.ElementTree as ET

from training.onnx.collect_native_slur_evidence import native_candidates, note_pitch


def test_native_evidence_retains_cross_voice_and_duplicate_endpoints():
    root = ET.fromstring("""<museScore><Score><Staff id="2"><Measure>
    <voice><Chord><Note><pitch>61</pitch></Note>
    <Spanner type="Slur"><prev><location><voices>2</voices><staves>1</staves></location></prev></Spanner>
    <Spanner type="Slur"><prev><location><voices>2</voices><staves>1</staves></location></prev></Spanner>
    </Chord></voice></Measure></Staff></Score></museScore>""")
    rows = native_candidates(root, 1, 61)
    assert len(rows) == 1
    assert rows[0]["pitch_matches"] and rows[0]["exact_measure"]
    chord = ET.fromstring(rows[0]["chord_xml"])
    assert len(chord.findall("Spanner")) == 2
    assert chord.findtext("Spanner/prev/location/voices") == "2"


def test_pitch_evidence_keeps_alteration_and_rest_distinct():
    note = ET.fromstring(
        "<note><pitch><step>B</step><alter>-1</alter><octave>6</octave></pitch></note>"
    )
    assert note_pitch(note) == 94
    assert note_pitch(ET.fromstring("<note><rest/></note>")) is None


def test_measure_neighbor_is_not_exact_identity_and_non_slur_is_ignored():
    root = ET.fromstring("""<museScore><Score><Staff id="3"><Measure>
    <Chord><Note><pitch>64</pitch></Note><Slur type="start" id="33"/></Chord>
    <Chord><Note><pitch>64</pitch></Note><Spanner type="Tie"/></Chord>
    </Measure><Measure/><Measure/><Measure/></Staff></Score></museScore>""")
    rows = native_candidates(root, 2, 64)
    assert len(rows) == 1 and not rows[0]["exact_measure"]
    assert native_candidates(root, 4, 64) == []
