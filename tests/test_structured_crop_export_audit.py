import unittest
import xml.etree.ElementTree as ET

from homr.transformer.structured_notation import (
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.audit_structured_crop_export import (
    endpoint_differences,
    non_span_hash,
    note_inventory,
    phrase_spans,
    render,
    span_differences,
)


def score(notes):
    return ET.fromstring(
        f"<score-partwise><part><measure>{notes}</measure></part></score-partwise>"
    )


class TestStructuredCropExportAudit(unittest.TestCase):
    def test_unmatched_boundary_stop_does_not_steal_another_voice_span(self):
        symbols = []
        for event, pitch, voice, kind in [
            ("a", "C5", "1", SlurEvent.START),
            ("b", "D5", "2", SlurEvent.STOP),
            ("c", "E5", "1", SlurEvent.STOP),
        ]:
            symbol = EncodedSymbol("note_4", pitch, position="upper")
            slots = list(empty_slur_slots())
            slots[0] = (kind, SlurSide.UNSPECIFIED)
            symbol.notation = NoteNotation(empty_beam_levels(), StemDirection.UNKNOWN, tuple(slots))
            symbol._onset_audit_id = event
            symbol._oracle_source_voice = (1, voice)
            symbols.append(symbol)
        fallback, _ = phrase_spans(render(symbols, arm="oracle-voice-owned"), source=False)
        strict, findings = phrase_spans(render(symbols, arm="oracle-identity-strict"), source=False)
        self.assertEqual(fallback, [("a", "b")])
        self.assertEqual(strict, [("a", "c")])
        self.assertEqual(findings, [{"kind": "unmatched_stop", "event": "b"}])

    def test_source_numbering_scope_conflict_remains_visible(self):
        xml = score(
            '<note data-onset-audit-id="a"><voice>1</voice><notations>'
            '<slur type="start" number="1"/></notations></note>'
            '<note data-onset-audit-id="b"><voice>2</voice><notations>'
            '<slur type="stop" number="1"/></notations></note>'
        )
        voice_spans, findings = phrase_spans(xml, source=True)
        part_spans, _ = phrase_spans(xml, source=True, source_scope="part")
        self.assertEqual(voice_spans, [])
        self.assertEqual(part_spans, [("a", "b")])
        self.assertEqual({f["kind"] for f in findings}, {"unmatched_stop", "unclosed_start"})

    def test_voice_oracle_isolates_same_staff_slot_collision(self):
        symbols = []
        for event, pitch, voice, kind in [
            ("a", "C5", "1", SlurEvent.START),
            ("b", "D5", "2", SlurEvent.START),
            ("c", "E5", "1", SlurEvent.STOP),
            ("d", "F5", "2", SlurEvent.STOP),
        ]:
            symbol = EncodedSymbol("note_4", pitch, position="upper")
            slots = list(empty_slur_slots())
            slots[0] = (kind, SlurSide.UNSPECIFIED)
            symbol.notation = NoteNotation(empty_beam_levels(), StemDirection.UNKNOWN, tuple(slots))
            symbol._onset_audit_id = event
            symbol._oracle_source_voice = (1, voice)
            symbols.append(symbol)
        current, _ = phrase_spans(render(symbols, arm="oracle-owned"), source=False)
        diagnostic, _ = phrase_spans(render(symbols, arm="oracle-voice-owned"), source=False)
        self.assertNotEqual(current, [("a", "c"), ("b", "d")])
        self.assertEqual(diagnostic, [("a", "c"), ("b", "d")])

    def test_tie_is_not_counted_as_phrase_slur(self):
        source = score(
            '<note data-onset-audit-id="a"><tie type="start"/>'
            '<notations><tied type="start"/></notations></note>'
        )
        output = score(
            '<note data-original-source-id="a">'
            '<notations><slur type="start"/></notations></note>'
        )
        differences = endpoint_differences(
            note_inventory(source, source=True), note_inventory(output, source=False), {"a"}
        )
        self.assertEqual(set(differences), {"phrase", "tied", "sounding_tie"})

    def test_identical_endpoint_counts_can_hide_wrong_concurrent_pairing(self):
        def xml(source, stop_numbers):
            attr = "data-onset-audit-id" if source else "data-original-source-id"
            return score(
                "".join(
                    f'<note {attr}="{event}"><voice>1</voice><notations>'
                    f'<slur type="{kind}" number="{number}"/></notations></note>'
                    for event, kind, number in [
                        ("a", "start", 1),
                        ("b", "start", 2),
                        ("c", "stop", stop_numbers[0]),
                        ("d", "stop", stop_numbers[1]),
                    ]
                )
            )

        source, output = xml(True, (2, 1)), xml(False, (1, 2))
        self.assertEqual(
            endpoint_differences(
                note_inventory(source, source=True),
                note_inventory(output, source=False),
                {"a", "b", "c", "d"},
            ),
            {},
        )
        expected, _ = phrase_spans(source, source=True)
        actual, _ = phrase_spans(output, source=False)
        comparison = span_differences(expected, actual, {"a", "b", "c", "d"})
        self.assertEqual(comparison["matched_interior_spans"], 0)
        self.assertEqual(len(comparison["missing_interior_spans"]), 2)

    def test_source_voices_have_independent_number_spaces(self):
        xml = score(
            "".join(
                f'<note data-onset-audit-id="{event}"><voice>{voice}</voice><notations>'
                f'<slur type="{kind}" number="1"/></notations></note>'
                for event, voice, kind in [
                    ("a", 1, "start"),
                    ("b", 2, "start"),
                    ("c", 1, "stop"),
                    ("d", 2, "stop"),
                ]
            )
        )
        spans, findings = phrase_spans(xml, source=True)
        self.assertEqual(spans, [("a", "c"), ("b", "d")])
        self.assertEqual(findings, [])
        comparison = span_differences(spans, [], {"c"})
        self.assertEqual(comparison["expected_interior_spans"], 0)
        self.assertEqual(comparison["source_boundary_spans"], [("a", "c")])

    def test_articulation_ownership_and_rhythmic_control_are_independent(self):
        source = score(
            '<note data-onset-audit-id="a"><duration>1</duration>'
            "<notations><articulations><accent/></articulations></notations></note>"
            '<note data-onset-audit-id="b"><notations/></note>'
        )
        output = score(
            '<note data-original-source-id="a"><duration>1</duration>'
            '<notations/></note><note data-original-source-id="b"><notations>'
            "<articulations><accent/></articulations></notations></note>"
        )
        differences = endpoint_differences(
            note_inventory(source, source=True), note_inventory(output, source=False), {"a", "b"}
        )
        self.assertEqual(len(differences["articulations"]), 2)
        baseline = score("<note><duration>1</duration><notations/></note>")
        tied = score(
            '<note><duration>1</duration><tie type="start"/>'
            '<notations><tied type="start"/><slur type="start"/></notations></note>'
        )
        self.assertEqual(non_span_hash(baseline), non_span_hash(tied))
        tied.find("./part/measure/note/duration").text = "2"
        self.assertNotEqual(non_span_hash(baseline), non_span_hash(tied))
