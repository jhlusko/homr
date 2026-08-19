import unittest

from homr.cross_staff_consistency import (
    Finding,
    analyze_system,
    check_clefs_against_profile,
    check_dangling_slurs,
    check_key_signatures,
    check_measure_counts,
    check_time_signatures,
    measure_count,
)
from homr.score_profile import ScorePart
from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
)
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


def _note_with_slur(slot: int, event: SlurEvent) -> EncodedSymbol:
    slots = [(SlurEvent.NONE, SlurSide.UNSPECIFIED)] * 6
    slots[slot] = (event, SlurSide.UNSPECIFIED)
    notation = NoteNotation(
        beam_levels=empty_beam_levels(),
        stem=StemDirection.NOT_APPLICABLE,
        slurs=tuple(slots),
    )
    symbol = EncodedSymbol("note_4")
    symbol.notation = notation
    return symbol


class TestMeasureCount(unittest.TestCase):
    def test_counts_every_barline_kind(self) -> None:
        symbols = [_sym("note_4"), _sym("barline"), _sym("note_4"), _sym("doublebarline")]

        self.assertEqual(measure_count(symbols), 2)

    def test_no_barlines_is_zero(self) -> None:
        self.assertEqual(measure_count([_sym("note_4")]), 0)


class TestMeasureCountConsistency(unittest.TestCase):
    def test_agreeing_staves_produce_no_finding(self) -> None:
        staff = [_sym("note_4"), _sym("barline")]

        self.assertEqual(check_measure_counts([staff, list(staff)]), [])

    def test_a_disagreement_is_reported_with_every_staff_named(self) -> None:
        four_four = [_sym("note_4"), _sym("barline")] * 4
        seven_eight = [_sym("note_4"), _sym("barline")] * 3

        findings = check_measure_counts([four_four, seven_eight, list(four_four)])

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "measure_count_mismatch")
        self.assertEqual(findings[0].staff_indices, (0, 1, 2))


class TestKeyAndTimeSignatures(unittest.TestCase):
    def test_matching_key_sequences_produce_no_finding(self) -> None:
        staff = [_sym("keySignature_-2"), _sym("note_4")]

        self.assertEqual(check_key_signatures([staff, list(staff)]), [])

    def test_a_different_key_is_reported(self) -> None:
        findings = check_key_signatures(
            [[_sym("keySignature_-2")], [_sym("keySignature_0")]]
        )

        self.assertEqual(findings[0].kind, "key_signature_mismatch")

    def test_a_later_key_change_missing_from_one_staff_is_still_a_mismatch(self) -> None:
        # Both staves agree on the opening key; only one of them also modulates. Looking
        # only at the first token would miss this.
        with_change = [_sym("keySignature_0"), _sym("note_4"), _sym("keySignature_-1")]
        without_change = [_sym("keySignature_0"), _sym("note_4")]

        findings = check_key_signatures([with_change, without_change])

        self.assertEqual(len(findings), 1)

    def test_matching_time_signatures_produce_no_finding(self) -> None:
        staff = [_sym("timeSignature/4"), _sym("note_4")]

        self.assertEqual(check_time_signatures([staff, list(staff)]), [])

    def test_a_different_time_signature_is_reported(self) -> None:
        findings = check_time_signatures(
            [[_sym("timeSignature/4")], [_sym("timeSignature/8")]]
        )

        self.assertEqual(findings[0].kind, "time_signature_mismatch")


class TestClefsAgainstProfile(unittest.TestCase):
    def test_a_clef_the_profile_expects_produces_no_finding(self) -> None:
        viola = ScorePart("viola", likely_clefs=("C3", "G2"))
        staff = [_sym("clef_C3"), _sym("note_4")]

        findings = check_clefs_against_profile([staff], {0: viola})

        self.assertEqual(findings, [])

    def test_an_unexpected_clef_is_reported(self) -> None:
        violin = ScorePart("violin-1", likely_clefs=("G2",))
        staff = [_sym("clef_F4"), _sym("note_4")]

        findings = check_clefs_against_profile([staff], {0: violin})

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "clef_profile_mismatch")
        self.assertEqual(findings[0].staff_indices, (0,))

    def test_a_staff_with_no_proposed_part_is_silent(self) -> None:
        staff = [_sym("clef_F4")]

        self.assertEqual(check_clefs_against_profile([staff], {}), [])

    def test_a_part_with_no_stated_clefs_is_silent(self) -> None:
        # An empty expectation means "we do not know," not "nothing is valid."
        unknown = ScorePart("mystery", likely_clefs=())
        staff = [_sym("clef_F4")]

        self.assertEqual(check_clefs_against_profile([staff], {0: unknown}), [])


class TestDanglingSlurs(unittest.TestCase):
    def test_a_matched_start_and_stop_produce_no_finding(self) -> None:
        staff = [_note_with_slur(0, SlurEvent.START), _note_with_slur(0, SlurEvent.STOP)]

        self.assertEqual(check_dangling_slurs([staff]), [])

    def test_a_start_never_closed_is_reported(self) -> None:
        staff = [_note_with_slur(0, SlurEvent.START)]

        findings = check_dangling_slurs([staff])

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "dangling_slur_start")

    def test_a_stop_with_nothing_open_is_reported(self) -> None:
        staff = [_note_with_slur(0, SlurEvent.STOP)]

        findings = check_dangling_slurs([staff])

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "dangling_slur_stop")

    def test_start_and_stop_on_one_note_closes_a_span_and_opens_a_new_one(self) -> None:
        staff = [
            _note_with_slur(0, SlurEvent.START),
            _note_with_slur(0, SlurEvent.START_AND_STOP),
            _note_with_slur(0, SlurEvent.STOP),
        ]

        self.assertEqual(check_dangling_slurs([staff]), [])

    def test_symbols_with_no_notation_at_all_are_not_flagged(self) -> None:
        staff = [_sym("note_4"), _sym("barline")]

        self.assertEqual(check_dangling_slurs([staff]), [])


class TestAnalyzeSystem(unittest.TestCase):
    def test_an_agreeing_system_has_no_findings(self) -> None:
        staff = [_sym("keySignature_0"), _sym("timeSignature/4"), _sym("note_4"), _sym("barline")]

        self.assertEqual(analyze_system([staff, list(staff)]), [])

    def test_findings_from_every_check_are_pooled(self) -> None:
        a = [_sym("keySignature_0"), _sym("note_4"), _sym("barline")]
        b = [_sym("keySignature_-1"), _sym("note_4")]

        findings = analyze_system([a, b])

        kinds = {f.kind for f in findings}
        self.assertIn("measure_count_mismatch", kinds)
        self.assertIn("key_signature_mismatch", kinds)

    def test_the_clef_check_only_runs_when_a_part_mapping_is_supplied(self) -> None:
        staff = [_sym("clef_F4")]

        findings = analyze_system([staff])

        self.assertNotIn("clef_profile_mismatch", {f.kind for f in findings})


if __name__ == "__main__":
    unittest.main()
