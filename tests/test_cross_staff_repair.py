import unittest

from homr.cross_staff_repair import (
    RepairProposal,
    apply_proposal,
    propose_majority_correction,
    propose_repairs,
)
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


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


if __name__ == "__main__":
    unittest.main()
