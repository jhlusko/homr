"""Native lyric identities retain timing, part ownership and structural text."""

import xml.etree.ElementTree as ET

from training.onnx.audit_lyric_reference_readiness import native_events


def test_polyphony_and_chords_keep_exact_native_ownership():
    root = ET.fromstring("""<score-partwise><part id="Voice1"><measure>
      <attributes><divisions>2</divisions></attributes>
      <note><pitch><step>G</step><alter>-1</alter><octave>4</octave></pitch>
        <duration>2</duration><voice>1</voice><lyric number="2"><text>du</text></lyric></note>
      <note><chord/><pitch><step>B</step><octave>4</octave></pitch>
        <duration>2</duration><voice>1</voice><lyric number="1"><text>her</text></lyric></note>
      <backup><duration>2</duration></backup>
      <note><pitch><step>C</step><octave>4</octave></pitch>
        <duration>1</duration><voice>2</voice><lyric><text>ge</text></lyric></note>
    </measure></part><part id="Voice2"><measure><note>
      <pitch><step>G</step><octave>4</octave></pitch><duration>1</duration>
      <lyric><text>ton</text></lyric></note></measure></part></score-partwise>""")
    events = native_events(root)
    assert [e["onset_quarters"] for e in events] == ["0", "0", "0", "0"]
    assert [e["duration_quarters"] for e in events] == ["1", "1", "1/2", "1"]
    assert [e["source_note_ordinal"] for e in events] == [1, 2, 3, 1]
    assert [e["source_part"] for e in events] == ["Voice1"] * 3 + ["Voice2"]
    assert events[0]["alter"] == "-1" and events[0]["verse"] == "2"
    assert events[2]["voice"] == "2"
    assert all(e["pixel_link_certified"] is False for e in events)


def test_elision_and_syllabic_metadata_are_preserved():
    root = ET.fromstring("""<score-partwise><part id="P1"><measure><note>
      <pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>
      <lyric><syllabic>begin</syllabic><text>a</text><elision> </elision>
        <text>b</text></lyric><lyric number="2"><extend/></lyric>
    </note></measure></part></score-partwise>""")
    events = native_events(root)
    assert len(events) == 1
    assert events[0]["syllabic"] == "begin"
    assert events[0]["text"] == "a b"
    assert events[0]["segments"][2] == {"kind": "elision", "text": " ", "attributes": {}}
