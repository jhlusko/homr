"""Findings are collectable as data, so a reviewer can be shown them.

They have always been written with `eprint`, which in a deployment is a container's
stdout - so a profile that correctly caught a wrong clef and a profile that was simply
wrong about the piece produced the same thing: nothing anyone sees.
"""

import unittest
from unittest.mock import patch

from homr.cross_staff_findings import collect_findings, record
from homr.staff_parsing import _report_cross_staff_findings
from homr.transformer.vocabulary import EncodedSymbol


class _Plan:
    def __init__(self, systems: int = 1) -> None:
        self.systems = list(range(systems))
        self.slots = [[0, 1, 2, 3] for _ in range(systems)]

    def staff_for_voice(self, system: int, voice: int) -> int | None:
        return voice


def _staff(*rhythms: str) -> list[EncodedSymbol]:
    return [EncodedSymbol(rhythm=r, pitch="C4") for r in rhythms]


class TestCollector(unittest.TestCase):
    def test_recording_without_a_collector_is_a_no_op(self) -> None:
        record("kind", "message")  # must not raise, must not accumulate
        with collect_findings() as found:
            pass
        self.assertEqual(found, [])

    def test_a_collector_receives_what_is_recorded(self) -> None:
        with collect_findings() as found:
            record("clef_profile_mismatch", "staff 3 decoded G2", system=1,
                   staff_indices=(3,), part="cello")
        self.assertEqual(
            found,
            [{
                "kind": "clef_profile_mismatch",
                "message": "staff 3 decoded G2",
                "system": 1,
                "staffIndices": [3],
                "part": "cello",
            }],
        )

    def test_optional_fields_are_omitted_rather_than_nulled(self) -> None:
        with collect_findings() as found:
            record("k", "m")
        self.assertEqual(found, [{"kind": "k", "message": "m"}])

    def test_the_sink_is_restored_after_the_block(self) -> None:
        with collect_findings() as outer:
            with collect_findings() as inner:
                record("k", "inner")
            record("k", "outer")
        self.assertEqual([f["message"] for f in inner], ["inner"])
        self.assertEqual([f["message"] for f in outer], ["outer"])


class TestReporterRecordsWhatItLogs(unittest.TestCase):
    def test_a_real_disagreement_is_collected_and_still_logged(self) -> None:
        # Three staves agreeing on 4/4 and one on 3/4: a time-signature mismatch.
        voices = [
            _staff("clef_G2", "timeSignature_4/4", "note_4", "barline"),
            _staff("clef_G2", "timeSignature_4/4", "note_4", "barline"),
            _staff("clef_G2", "timeSignature_4/4", "note_4", "barline"),
            _staff("clef_F4", "timeSignature_3/4", "note_4", "barline"),
        ]
        with patch("homr.staff_parsing.eprint") as logged:
            with collect_findings() as found:
                _report_cross_staff_findings(_Plan(), voices, None)

        kinds = {f["kind"] for f in found}
        self.assertIn("time_signature_mismatch", kinds)
        # Still logged: this adds a sink, it does not replace the log.
        self.assertTrue(logged.called)
        for finding in found:
            self.assertIn("message", finding)
            self.assertTrue(finding["message"])

    def test_agreeing_staves_produce_no_findings(self) -> None:
        voices = [_staff("clef_G2", "timeSignature_4/4", "note_4", "barline")] * 4
        with patch("homr.staff_parsing.eprint"):
            with collect_findings() as found:
                _report_cross_staff_findings(_Plan(), voices, None)
        self.assertEqual([f for f in found if f["kind"] == "time_signature_mismatch"], [])


if __name__ == "__main__":
    unittest.main()
