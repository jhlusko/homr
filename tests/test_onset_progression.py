import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.audit_onset_progression import trace_progression


class TestOnsetProgression(unittest.TestCase):
    def test_overlapping_voice_requires_gap_shorter_than_own_duration(self):
        upper = EncodedSymbol("note_4", "C5", position="upper")
        lower = EncodedSymbol("note_8", "C3", position="lower")
        upper._onset_audit_id = "upper"
        lower._onset_audit_id = "lower"
        events = {
            "upper": {"part": 1, "physical_measure": 1, "onset_quarters": "1/2", "grace": False},
            "lower": {"part": 1, "physical_measure": 1, "onset_quarters": "1", "grace": False},
        }
        rows = trace_progression([upper, lower], events)
        self.assertEqual(rows[0]["legacy_advance_quarters"], "1")
        self.assertEqual(rows[0]["source_next_onset_gap_quarters"], "1/2")
        self.assertTrue(rows[0]["advance_mismatch"])
        self.assertEqual(str(upper.get_duration().fraction), "1/4")

    def test_measure_boundary_is_not_treated_as_negative_onset_gap(self):
        first, last = EncodedSymbol("note_4", "C5"), EncodedSymbol("note_4", "D5")
        first._onset_audit_id, last._onset_audit_id = "a", "b"
        events = {
            "a": {"part": 1, "physical_measure": 1, "onset_quarters": "3", "grace": False},
            "b": {"part": 1, "physical_measure": 2, "onset_quarters": "0", "grace": False},
        }
        rows = trace_progression([first, EncodedSymbol("barline"), last], events)
        self.assertNotIn("source_next_onset_gap_quarters", rows[0])
