import unittest
from fractions import Fraction

from homr.cross_staff_repair import (
    ArticulationRepairProposal,
    InsertionRepairProposal,
    RepairProposal,
    apply_articulation_proposal,
    apply_insertion_proposal,
    apply_proposal,
    propose_carry_forward_key_signature,
    propose_majority_correction,
    propose_majority_position_corrections,
    propose_motif_articulation_corrections,
    propose_repairs,
)
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


def _note(rhythm: str, pitch: str, articulation: str = "_") -> EncodedSymbol:
    return EncodedSymbol(rhythm=rhythm, pitch=pitch, articulation=articulation)


class TestProposeMajorityCorrection(unittest.TestCase):
    def test_agreeing_staves_produce_no_proposal(self) -> None:
        staff = [_sym("clef_G2"), _sym("keySignature_0"), _sym("note_4")]

        proposals = propose_majority_correction([staff, list(staff), list(staff)], "keySignature")

        self.assertEqual(proposals, [])

    def test_the_minority_staff_gets_a_proposal_toward_the_majority(self) -> None:
        majority = [_sym("timeSignature/4"), _sym("note_4")]
        minority = [_sym("timeSignature/8"), _sym("note_4")]

        proposals = propose_majority_correction(
            [list(majority), list(majority), list(majority), list(minority)], "timeSignature"
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].staff_index, 3)
        self.assertEqual(proposals[0].position, 0)
        self.assertEqual(proposals[0].current_rhythm, "timeSignature/8")
        self.assertEqual(proposals[0].proposed_rhythm, "timeSignature/4")

    def test_every_minority_staff_gets_its_own_proposal(self) -> None:
        majority = [_sym("keySignature_0")]
        minority_a = [_sym("keySignature_-1")]
        minority_b = [_sym("keySignature_2")]

        proposals = propose_majority_correction(
            [list(majority), list(majority), list(minority_a), list(minority_b)], "keySignature"
        )

        self.assertEqual({p.staff_index for p in proposals}, {2, 3})
        self.assertTrue(all(p.proposed_rhythm == "keySignature_0" for p in proposals))

    def test_a_genuine_tie_proposes_nothing(self) -> None:
        a = [_sym("keySignature_0")]
        b = [_sym("keySignature_-1")]

        proposals = propose_majority_correction([list(a), list(b)], "keySignature")

        self.assertEqual(proposals, [])

    def test_fewer_than_two_staves_stating_a_value_proposes_nothing(self) -> None:
        staff = [_sym("keySignature_0")]
        no_key = [_sym("note_4")]

        proposals = propose_majority_correction([list(staff), list(no_key)], "keySignature")

        self.assertEqual(proposals, [])

    def test_only_the_opening_value_is_considered_not_later_changes(self) -> None:
        # Two staves open on 0 sharps; a third opens on 0 but modulates later - the
        # later change is not what this function evaluates.
        a = [_sym("keySignature_0"), _sym("note_4")]
        b = [_sym("keySignature_0"), _sym("note_4"), _sym("keySignature_-1")]

        proposals = propose_majority_correction([list(a), list(a), list(b)], "keySignature")

        self.assertEqual(proposals, [])

    def test_a_staff_with_no_matching_prefix_at_all_is_ignored(self) -> None:
        with_key = [_sym("keySignature_0")]
        no_key = [_sym("note_4"), _sym("barline")]

        proposals = propose_majority_correction(
            [list(with_key), list(with_key), list(no_key)], "keySignature"
        )

        self.assertEqual(proposals, [])


class TestProposeCarryForwardKeySignature(unittest.TestCase):
    def test_a_lone_stated_value_is_carried_to_every_silent_staff(self) -> None:
        # The exact real-page case this rule exists for: one staff states it, the rest
        # are silent, and no staff states a conflicting value.
        stating = [_sym("clef_C3"), _sym("keySignature_3"), _sym("note_4")]
        silent = [_sym("clef_G2"), _sym("note_4")]

        proposals = propose_carry_forward_key_signature(
            [list(silent), list(silent), list(stating), list(silent)]
        )

        self.assertEqual(len(proposals), 3)
        self.assertEqual({p.staff_index for p in proposals}, {0, 1, 3})
        self.assertTrue(all(p.inserted_rhythm == "keySignature_3" for p in proposals))

    def test_inserts_after_a_leading_clef_not_before_it(self) -> None:
        stating = [_sym("keySignature_2"), _sym("note_4")]
        silent = [_sym("clef_F4"), _sym("note_4")]

        proposals = propose_carry_forward_key_signature([list(stating), list(silent)])

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].position, 1)

    def test_inserts_at_position_zero_with_no_leading_clef(self) -> None:
        stating = [_sym("keySignature_2"), _sym("note_4")]
        silent = [_sym("note_4")]

        proposals = propose_carry_forward_key_signature([list(stating), list(silent)])

        self.assertEqual(proposals[0].position, 0)

    def test_a_conflicting_stated_value_proposes_nothing(self) -> None:
        # A genuine disagreement is propose_majority_correction's territory, not this
        # rule's - refusing here rather than guessing between the two mechanisms.
        a = [_sym("keySignature_0"), _sym("note_4")]
        b = [_sym("keySignature_-1"), _sym("note_4")]
        silent = [_sym("note_4")]

        proposals = propose_carry_forward_key_signature([list(a), list(b), list(silent)])

        self.assertEqual(proposals, [])

    def test_no_staff_stating_anything_proposes_nothing(self) -> None:
        silent = [_sym("note_4")]

        proposals = propose_carry_forward_key_signature([list(silent), list(silent)])

        self.assertEqual(proposals, [])

    def test_every_staff_already_stating_the_same_value_proposes_nothing(self) -> None:
        staff = [_sym("keySignature_0"), _sym("note_4")]

        proposals = propose_carry_forward_key_signature([list(staff), list(staff)])

        self.assertEqual(proposals, [])

    def test_a_single_staff_system_has_nothing_silent_to_fix(self) -> None:
        staff = [_sym("keySignature_0"), _sym("note_4")]

        self.assertEqual(propose_carry_forward_key_signature([staff]), [])


class TestApplyInsertionProposal(unittest.TestCase):
    def test_inserts_before_the_given_position(self) -> None:
        staff = [_sym("clef_G2"), _sym("note_4")]
        proposal = InsertionRepairProposal(
            staff_index=0, position=1, inserted_rhythm="keySignature_3", reason="test"
        )

        corrected = apply_insertion_proposal(staff, proposal)

        self.assertEqual([s.rhythm for s in corrected], ["clef_G2", "keySignature_3", "note_4"])

    def test_the_original_list_is_not_mutated(self) -> None:
        staff = [_sym("clef_G2"), _sym("note_4")]
        proposal = InsertionRepairProposal(0, 1, "keySignature_3", "test")

        apply_insertion_proposal(staff, proposal)

        self.assertEqual([s.rhythm for s in staff], ["clef_G2", "note_4"])

    def test_position_zero_inserts_at_the_very_start(self) -> None:
        staff = [_sym("note_4")]
        proposal = InsertionRepairProposal(0, 0, "keySignature_0", "test")

        corrected = apply_insertion_proposal(staff, proposal)

        self.assertEqual([s.rhythm for s in corrected], ["keySignature_0", "note_4"])

    def test_position_equal_to_length_appends_at_the_end(self) -> None:
        staff = [_sym("note_4")]
        proposal = InsertionRepairProposal(0, 1, "keySignature_0", "test")

        corrected = apply_insertion_proposal(staff, proposal)

        self.assertEqual([s.rhythm for s in corrected], ["note_4", "keySignature_0"])

    def test_an_out_of_range_position_is_refused(self) -> None:
        staff = [_sym("note_4")]
        proposal = InsertionRepairProposal(0, 5, "keySignature_0", "test")

        with self.assertRaises(ValueError):
            apply_insertion_proposal(staff, proposal)

    def test_the_staff_grows_by_exactly_one_symbol(self) -> None:
        staff = [_sym("clef_G2"), _sym("note_4"), _sym("barline")]
        proposal = InsertionRepairProposal(0, 1, "keySignature_3", "test")

        corrected = apply_insertion_proposal(staff, proposal)

        self.assertEqual(len(corrected), len(staff) + 1)


class TestApplyProposal(unittest.TestCase):
    def test_replaces_exactly_the_proposed_position(self) -> None:
        staff = [_sym("timeSignature/8"), _sym("note_4")]
        proposal = RepairProposal(
            staff_index=0,
            position=0,
            current_rhythm="timeSignature/8",
            proposed_rhythm="timeSignature/4",
            reason="test",
        )

        corrected = apply_proposal(staff, proposal)

        self.assertEqual(corrected[0].rhythm, "timeSignature/4")
        self.assertEqual(corrected[1].rhythm, "note_4")
        self.assertEqual(len(corrected), len(staff))

    def test_the_original_list_is_not_mutated(self) -> None:
        staff = [_sym("timeSignature/8")]
        proposal = RepairProposal(0, 0, "timeSignature/8", "timeSignature/4", "test")

        apply_proposal(staff, proposal)

        self.assertEqual(staff[0].rhythm, "timeSignature/8")

    def test_a_stale_proposal_is_refused(self) -> None:
        staff = [_sym("timeSignature/4")]  # already corrected by something else
        proposal = RepairProposal(0, 0, "timeSignature/8", "timeSignature/4", "test")

        with self.assertRaises(ValueError):
            apply_proposal(staff, proposal)

    def test_an_out_of_range_position_is_refused(self) -> None:
        staff = [_sym("timeSignature/8")]
        proposal = RepairProposal(0, 5, "timeSignature/8", "timeSignature/4", "test")

        with self.assertRaises(ValueError):
            apply_proposal(staff, proposal)

    def test_non_rhythm_fields_are_preserved(self) -> None:
        original = EncodedSymbol(
            rhythm="note_4", pitch="C5", lift="#", articulation="staccato", slur="slurStart"
        )
        proposal = RepairProposal(0, 0, "note_4", "note_8", "test")

        corrected = apply_proposal([original], proposal)

        self.assertEqual(corrected[0].pitch, "C5")
        self.assertEqual(corrected[0].lift, "#")
        self.assertEqual(corrected[0].articulation, "staccato")
        self.assertEqual(corrected[0].slur, "slurStart")


class TestProposeRepairs(unittest.TestCase):
    def test_pools_both_key_and_time_signature_proposals(self) -> None:
        majority = [_sym("keySignature_0"), _sym("timeSignature/4")]
        minority = [_sym("keySignature_-1"), _sym("timeSignature/8")]

        proposals = propose_repairs([list(majority), list(majority), list(minority)])

        current = {p.current_rhythm for p in proposals}
        self.assertEqual(len(proposals), 2)
        self.assertEqual(current, {"keySignature_-1", "timeSignature/8"})

    def test_agreeing_staves_propose_nothing(self) -> None:
        staff = [_sym("keySignature_0"), _sym("timeSignature/4")]

        proposals = propose_repairs([list(staff), list(staff), list(staff)])

        self.assertEqual(proposals, [])


def _motif(articulation: str = "_") -> list:
    return [
        _note("note_4", "C5"),
        _note("note_4", "D5"),
        _note("note_4", "E5", articulation),
        _note("note_4", "F5"),
    ]


class TestProposeMotifArticulationCorrections(unittest.TestCase):
    def test_three_corroborating_staves_flag_the_minority(self) -> None:
        staves = [_motif("staccato"), _motif("staccato"), _motif("accent")]

        proposals = propose_motif_articulation_corrections(staves, min_motif_length=4)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].staff_index, 2)
        self.assertEqual(proposals[0].current_articulation, "accent")
        self.assertEqual(proposals[0].proposed_articulation, "staccato")

    def test_only_two_matching_staves_is_not_enough_evidence(self) -> None:
        # Two staves disagreeing on their own is a coin flip, not a majority - the
        # exact case check_shared_motifs' pairwise finding cannot resolve on its own.
        staves = [_motif("staccato"), _motif("accent")]

        self.assertEqual(propose_motif_articulation_corrections(staves), [])

    def test_all_staves_agreeing_proposes_nothing(self) -> None:
        staves = [_motif("staccato"), _motif("staccato"), _motif("staccato")]

        self.assertEqual(propose_motif_articulation_corrections(staves), [])

    def test_a_genuine_three_way_tie_proposes_nothing(self) -> None:
        staves = [_motif("staccato"), _motif("accent"), _motif("tenuto")]

        self.assertEqual(propose_motif_articulation_corrections(staves), [])

    def test_a_run_shorter_than_the_minimum_is_not_corroboration(self) -> None:
        short = [_note("note_4", "C5"), _note("note_4", "D5", "accent")]
        staves = [short, list(short), [_note("note_4", "C5"), _note("note_4", "D5")]]

        self.assertEqual(propose_motif_articulation_corrections(staves, min_motif_length=4), [])

    def test_a_group_is_reported_once_not_once_per_staff(self) -> None:
        staves = [_motif("staccato"), _motif("staccato"), _motif("accent")]

        proposals = propose_motif_articulation_corrections(staves, min_motif_length=4)

        # Only the minority staff needed a correction, and only one report of it.
        self.assertEqual(len(proposals), 1)

    def test_an_unrelated_fourth_staff_does_not_interfere(self) -> None:
        unrelated = [_note("note_8", "G3"), _note("note_8", "A3"), _note("note_8", "B3")]
        staves = [_motif("staccato"), _motif("staccato"), _motif("accent"), unrelated]

        proposals = propose_motif_articulation_corrections(staves, min_motif_length=4)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].staff_index, 2)


class TestApplyArticulationProposal(unittest.TestCase):
    def test_replaces_exactly_the_proposed_positions_articulation(self) -> None:
        staff = [_note("note_4", "C5", "accent"), _note("note_4", "D5")]
        proposal = ArticulationRepairProposal(
            staff_index=0, position=0, current_articulation="accent",
            proposed_articulation="staccato", reason="test",
        )

        corrected = apply_articulation_proposal(staff, proposal)

        self.assertEqual(corrected[0].articulation, "staccato")
        self.assertEqual(corrected[1].rhythm, "note_4")
        self.assertEqual(len(corrected), len(staff))

    def test_the_original_list_is_not_mutated(self) -> None:
        staff = [_note("note_4", "C5", "accent")]
        proposal = ArticulationRepairProposal(0, 0, "accent", "staccato", "test")

        apply_articulation_proposal(staff, proposal)

        self.assertEqual(staff[0].articulation, "accent")

    def test_a_stale_proposal_is_refused(self) -> None:
        staff = [_note("note_4", "C5", "staccato")]  # already corrected by something else
        proposal = ArticulationRepairProposal(0, 0, "accent", "staccato", "test")

        with self.assertRaises(ValueError):
            apply_articulation_proposal(staff, proposal)

    def test_an_out_of_range_position_is_refused(self) -> None:
        staff = [_note("note_4", "C5", "accent")]
        proposal = ArticulationRepairProposal(0, 5, "accent", "staccato", "test")

        with self.assertRaises(ValueError):
            apply_articulation_proposal(staff, proposal)

    def test_non_articulation_fields_are_preserved(self) -> None:
        original = EncodedSymbol(
            rhythm="note_4", pitch="C5", lift="#", articulation="accent", slur="slurStart"
        )
        proposal = ArticulationRepairProposal(0, 0, "accent", "staccato", "test")

        corrected = apply_articulation_proposal([original], proposal)

        self.assertEqual(corrected[0].pitch, "C5")
        self.assertEqual(corrected[0].lift, "#")
        self.assertEqual(corrected[0].rhythm, "note_4")
        self.assertEqual(corrected[0].slur, "slurStart")


def _measures(*measure_note_counts: int) -> list:
    """A staff of quarter-note measures, one `_sym("barline")` after each - e.g.
    `_measures(4, 4)` is two full 4/4 measures."""
    symbols = []
    for note_count in measure_note_counts:
        symbols.extend([_sym("note_4")] * note_count)
        symbols.append(_sym("barline"))
    return symbols


class TestProposeMajorityPositionCorrections(unittest.TestCase):
    def test_agreeing_staves_propose_nothing(self) -> None:
        staff = _measures(4, 4)

        proposals = propose_majority_position_corrections([list(staff) for _ in range(4)])

        self.assertEqual(proposals, [])

    def test_three_staff_majority_flags_the_lone_diverging_staff(self) -> None:
        majority = _measures(4, 4)
        # An extra quarter note in the first measure only - every later barline is
        # shifted by the same constant amount, never catching back up.
        diverging = _measures(5, 4)

        proposals = propose_majority_position_corrections(
            [list(majority), list(majority), list(majority), diverging]
        )

        self.assertEqual(len(proposals), 1)
        proposal = proposals[0]
        self.assertEqual(proposal.staff_index, 3)
        self.assertEqual(proposal.measure_index, 0)
        self.assertEqual(proposal.offset, Fraction(-1, 4))
        self.assertEqual(proposal.corroborating_staves, 3)
        self.assertIn("staff 3", proposal.reason)
        self.assertIn("measure 1", proposal.reason)

    def test_fewer_than_three_agreeing_staves_proposes_nothing(self) -> None:
        majority = _measures(4, 4)
        diverging = _measures(5, 4)

        proposals = propose_majority_position_corrections([majority, diverging])

        self.assertEqual(proposals, [])

    def test_a_tied_majority_proposes_nothing(self) -> None:
        sequence_a = _measures(4, 4)
        sequence_b = _measures(3, 4)

        proposals = propose_majority_position_corrections(
            [list(sequence_a)] * 3 + [list(sequence_b)] * 3
        )

        self.assertEqual(proposals, [])

    def test_a_non_constant_offset_declines_to_propose(self) -> None:
        majority = _measures(4, 4, 4)
        # Diverges by a different amount at each barline - a messier disagreement than
        # one localized mis-decode, so this should decline rather than guess.
        messy = _measures(5, 3, 6)

        proposals = propose_majority_position_corrections(
            [list(majority), list(majority), list(majority), messy]
        )

        self.assertEqual(proposals, [])


if __name__ == "__main__":
    unittest.main()
