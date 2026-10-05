import xml.etree.ElementTree as ET

from training.onnx.audit_notation_spans import compare, extract


def score(notes):
    return ET.fromstring('<score-partwise><part id="P1"><measure number="1">'
                         '<attributes><divisions>1</divisions></attributes>'
                         + notes + '</measure></part></score-partwise>')


def note(tags, voice=1):
    return ('<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration>'
            f'<voice>{voice}</voice>' + tags + '</note>')


def test_tie_chain_stops_before_start_independent_of_xml_tag_order():
    first = '<tie type="start"/><notations><tied type="start"/></notations>'
    middle = ('<tie type="start"/><tie type="stop"/>'
              '<notations><tied type="start"/><tied type="stop"/></notations>')
    last = '<tie type="stop"/><notations><tied type="stop"/></notations>'
    _, spans, issues = extract(score(note(first) + note(middle) + note(last)))
    assert sum(spans['sounding_tie'].values()) == 2
    assert spans['sounding_tie'] == spans['engraved_tie']
    assert issues == {}


def test_simultaneous_tie_and_phrase_remain_independent():
    first = ('<tie type="start"/><notations><tied type="start"/>'
             '<slur number="1" type="start"/></notations>')
    last = ('<tie type="stop"/><notations><tied type="stop"/>'
            '<slur number="1" type="stop"/></notations>')
    _, spans, issues = extract(score(note(first) + note(last)))
    assert all(sum(s.values()) == 1 for s in spans.values())
    assert issues == {}


def test_duplicate_phrase_start_excluded_instead_of_guessed():
    first = '<notations><slur type="start"/></notations>'
    last = '<notations><slur type="stop"/></notations>'
    _, spans, issues = extract(score(note(first) + note(first) + note(last)))
    assert not spans['phrase']
    assert issues['phrase:ambiguous_pair_excluded'] == 1


def test_boundary_span_is_excluded_and_reported():
    root = score(note('<notations><slur type="start"/></notations>'))
    part = root.find('part')
    part.append(ET.fromstring('<measure number="2">'
                             + note('<notations><slur type="stop"/></notations>')
                             + '</measure>'))
    _, spans, issues = extract(root, start=1, end=2)
    assert not spans['phrase']
    assert issues['phrase:boundary_or_nonpitched_endpoint'] == 1


def test_unmatched_endpoint_not_counted_as_correct_span():
    _, spans, _ = extract(score(note('<tie type="start"/>') + note('<tie type="stop"/>')))
    result = compare(spans, spans, set())
    assert result['sounding_tie']['matched'] == 0
    assert result['sounding_tie']['reference_excluded'] == 1
