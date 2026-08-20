import unittest

from homr.cross_staff_consistency import (
    analyze_system,
    check_clefs_against_profile,
    check_dangling_slurs,
    check_key_signatures,
    check_measure_counts,
    check_shared_motifs,
    check_time_signatures,
    findings_by_page,
    measure_count,
    split_by_system,
    staves_by_system,
)
from homr.score_profile import ScorePart
from homr.transformer.structured_notation import (
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
)
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


def _note(rhythm: str, pitch: str, articulation: str = "_") -> EncodedSymbol:
    return EncodedSymbol(rhythm=rhythm, pitch=pitch, articulation=articulation)


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


class TestSharedMotifs(unittest.TestCase):
    def test_a_matching_run_with_one_differing_articulation_is_reported(self) -> None:
        staff_a = [_note("note_4", "C5"), _note("note_4", "D5", "accent"), _note("note_4", "E5")]
        staff_b = [_note("note_4", "C5"), _note("note_4", "D5", "marcato"), _note("note_4", "E5")]

        findings = check_shared_motifs([staff_a, staff_b], min_motif_length=3)

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].kind, "motif_articulation_mismatch")
        self.assertEqual(findings[0].staff_indices, (0, 1))

    def test_an_identical_run_produces_no_finding(self) -> None:
        staff = [_note("note_4", "C5"), _note("note_4", "D5"), _note("note_4", "E5")]

        findings = check_shared_motifs([staff, list(staff)], min_motif_length=3)

        self.assertEqual(findings, [])

    def test_a_run_shorter_than_the_minimum_is_ignored(self) -> None:
        staff_a = [_note("note_4", "C5"), _note("note_4", "D5", "accent")]
        staff_b = [_note("note_4", "C5"), _note("note_4", "D5", "marcato")]

        findings = check_shared_motifs([staff_a, staff_b], min_motif_length=3)

        self.assertEqual(findings, [])

    def test_a_transposed_entry_is_not_matched(self) -> None:
        # A known, named limitation - matching is on absolute pitch, not interval.
        staff_a = [_note("note_4", "C5"), _note("note_4", "D5"), _note("note_4", "E5")]
        staff_b = [_note("note_4", "G5"), _note("note_4", "A5"), _note("note_4", "B5")]

        findings = check_shared_motifs([staff_a, staff_b], min_motif_length=3)

        self.assertEqual(findings, [])

    def test_non_note_symbols_do_not_break_the_alignment(self) -> None:
        staff_a = [
            _note("note_4", "C5"),
            EncodedSymbol("barline"),
            _note("note_4", "D5", "accent"),
            _note("note_4", "E5"),
        ]
        staff_b = [
            _note("note_4", "C5"),
            _note("note_4", "D5", "marcato"),
            EncodedSymbol("barline"),
            _note("note_4", "E5"),
        ]

        findings = check_shared_motifs([staff_a, staff_b], min_motif_length=3)

        self.assertEqual(len(findings), 1)

    def test_every_pair_of_staves_is_compared(self) -> None:
        matching = [_note("note_4", "C5"), _note("note_4", "D5", "accent"), _note("note_4", "E5")]
        different = [_note("note_4", "C5"), _note("note_4", "D5", "marcato"), _note("note_4", "E5")]
        unrelated = [_note("note_8", "F3"), _note("note_8", "G3"), _note("note_8", "A3")]

        findings = check_shared_motifs([matching, different, unrelated], min_motif_length=3)

        self.assertEqual({f.staff_indices for f in findings}, {(0, 1)})


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


class TestSplitBySystem(unittest.TestCase):
    def test_one_system_with_no_newline_is_one_chunk(self) -> None:
        symbols = [_sym("note_4"), _sym("barline")]

        self.assertEqual(split_by_system(symbols), [symbols])

    def test_a_trailing_newline_does_not_produce_an_empty_final_chunk(self) -> None:
        # parse_staffs appends "newline" after every staff, including a voice's last one.
        symbols = [_sym("note_4"), _sym("newline")]

        self.assertEqual(split_by_system(symbols), [[_sym("note_4")]])

    def test_multiple_systems_split_at_each_newline(self) -> None:
        symbols = [_sym("note_4"), _sym("newline"), _sym("note_8"), _sym("newline")]

        chunks = split_by_system(symbols)

        self.assertEqual(chunks, [[_sym("note_4")], [_sym("note_8")]])

    def test_an_empty_stream_produces_no_chunks(self) -> None:
        self.assertEqual(split_by_system([]), [])


class TestFindingsByPage(unittest.TestCase):
    def test_a_page_where_every_voice_appears_in_every_system(self) -> None:
        # Two voices, two systems, no gaps - the simple case findings_by_page must not
        # break on its way to handling the harder one below.
        voice_a = [
            _sym("keySignature_0"), _sym("newline"), _sym("keySignature_0"), _sym("newline")
        ]  # fmt: skip
        voice_b = [
            _sym("keySignature_0"), _sym("newline"), _sym("keySignature_-1"), _sym("newline")
        ]  # fmt: skip

        results = findings_by_page([voice_a, voice_b], [[True, True], [True, True]])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], [])  # system 1: both voices agree
        self.assertEqual(len(results[1]), 1)  # system 2: keys disagree
        self.assertEqual(results[1][0].kind, "key_signature_mismatch")

    def test_a_voice_missing_from_one_system_does_not_shift_the_other_systems_chunks(
        self,
    ) -> None:
        # Voice B is absent from system 2 (e.g. an incomplete system spacing recovered).
        # Voice A has three chunks (systems 1, 2, 3); voice B has only two (systems 1
        # and 3) - naive positional zipping would compare A's system-3 chunk against
        # B's system-2 chunk and misattribute every finding from system 3 onward.
        voice_a = (
            [_sym("keySignature_0"), _sym("newline")]
            + [_sym("keySignature_0"), _sym("newline")]
            + [_sym("keySignature_-3"), _sym("newline")]
        )
        voice_b = (
            [_sym("keySignature_0"), _sym("newline")] + [_sym("keySignature_-3"), _sym("newline")]
        )
        presence = [[True, True], [True, False], [True, True]]

        results = findings_by_page([voice_a, voice_b], presence)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], [])  # system 1: both present, both key 0
        self.assertEqual(len(results[1]), 0)  # system 2: only voice A - nothing to compare
        # system 3: both present, both key -3 - correctly matched despite the gap.
        self.assertEqual(results[2], [])

    def test_a_staff_to_part_mapping_is_forwarded_per_system(self) -> None:
        violin = ScorePart("violin-1", likely_clefs=("G2",))
        voice_a = [_sym("clef_F4"), _sym("newline")]

        results = findings_by_page([voice_a], [[True]], staff_to_part_by_system=[{0: violin}])

        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[0][0].kind, "clef_profile_mismatch")


class TestStavesBySystem(unittest.TestCase):
    def test_reshapes_voice_major_into_system_major(self) -> None:
        voice_a = [_sym("note_4"), _sym("newline"), _sym("note_8"), _sym("newline")]
        voice_b = [_sym("rest_4"), _sym("newline"), _sym("rest_8"), _sym("newline")]

        result = staves_by_system([voice_a, voice_b], [[True, True], [True, True]])

        self.assertEqual(len(result), 2)
        self.assertEqual([s.rhythm for s in result[0][0]], ["note_4"])
        self.assertEqual([s.rhythm for s in result[0][1]], ["rest_4"])
        self.assertEqual([s.rhythm for s in result[1][0]], ["note_8"])

    def test_matches_findings_by_pages_own_reshaping(self) -> None:
        # findings_by_page is now built directly on this function - this is a
        # regression guard against the two ever drifting apart again.
        voice_a = [
            _sym("keySignature_0"), _sym("newline"), _sym("keySignature_-1"), _sym("newline")
        ]  # fmt: skip
        presence = [[True], [True]]

        staves = staves_by_system([voice_a], presence)
        findings = findings_by_page([voice_a], presence)

        self.assertEqual(len(staves), len(findings))


if __name__ == "__main__":
    unittest.main()
