"""Experimental versioned timing sidecars for future corpus builds.

Independent of legacy rhythm tokens and checkpoint vocabulary. No current
builder, trainer or runtime consumes this format. Duration is exact; printed
type/dots, full-measure-rest status, time modification and nested tuplet
notations remain separate so a dotted alias cannot replace a duplet silently.
"""
from fractions import Fraction
import xml.etree.ElementTree as ET

SCHEMA = 'exact-timing-proposal-v1'


def _canonical_xml(element):
    # Ignore indentation and empty-element spelling in the serialized sidecar.
    return ET.canonicalize(ET.tostring(element, encoding='unicode'), strip_text=True)


def extract_exact_timing(note, divisions):
    divisions = Fraction(divisions)
    if divisions <= 0 or note.find('duration') is None or note.find('grace') is not None:
        raise ValueError('Requires a timed non-grace note and positive divisions')
    elapsed = Fraction(note.findtext('duration')) / divisions
    if elapsed <= 0:
        raise ValueError('Requires positive elapsed duration')
    modification = note.find('time-modification')
    rest = note.find('rest')
    return dict(schema=SCHEMA, elapsed_quarters=str(elapsed),
        written_type=note.findtext('type'), dots=len(note.findall('dot')),
        full_measure_rest=rest is not None and rest.get('measure') == 'yes',
        time_modification_xml=_canonical_xml(modification)
            if modification is not None else None,
        tuplet_notations_xml=[_canonical_xml(t)
            for t in note.findall('./notations/tuplet')])


def apply_exact_timing(note, payload, divisions):
    """Apply a sidecar to an existing note shell, preserving unrelated fields.

    Caller owns note ordering/voices and MusicXML child ordering. This audit
    adapter is not the production score exporter. Refuses lossy tick rounding.
    """
    if payload['schema'] != SCHEMA:
        raise ValueError('Unknown timing schema')
    elapsed = Fraction(payload['elapsed_quarters'])
    divisions = Fraction(divisions)
    ticks = elapsed * divisions
    if elapsed <= 0 or divisions <= 0 or ticks.denominator != 1:
        raise ValueError('Duration cannot be represented exactly at these divisions')
    if note.find('grace') is not None:
        raise ValueError('Cannot apply timed sidecar to a grace note')
    duration = note.find('duration')
    if duration is None:
        duration = ET.SubElement(note, 'duration')
    duration.text = str(ticks.numerator)
    for tag in ('type', 'dot', 'time-modification'):
        for child in note.findall(tag):
            note.remove(child)
    if payload['written_type'] is not None:
        ET.SubElement(note, 'type').text = payload['written_type']
    for _ in range(payload['dots']):
        ET.SubElement(note, 'dot')
    rest = note.find('rest')
    if payload['full_measure_rest']:
        if rest is None:
            raise ValueError('Full-measure-rest payload requires a rest shell')
        rest.set('measure', 'yes')
    elif rest is not None:
        rest.attrib.pop('measure', None)
    if payload['time_modification_xml'] is not None:
        element = ET.fromstring(payload['time_modification_xml'])
        if element.tag != 'time-modification':
            raise ValueError('Invalid time-modification payload')
        note.append(element)
    for notations in note.findall('notations'):
        for tuplet in notations.findall('tuplet'):
            notations.remove(tuplet)
    if payload['tuplet_notations_xml']:
        notations = note.find('notations')
        if notations is None:
            notations = ET.SubElement(note, 'notations')
        for text in payload['tuplet_notations_xml']:
            tuplet = ET.fromstring(text)
            if tuplet.tag != 'tuplet':
                raise ValueError('Invalid tuplet notation payload')
            notations.append(tuplet)
