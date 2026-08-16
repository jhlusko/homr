import unittest

from homr.transformer.beam_validation import validate_level, validate_voice
from homr.transformer.structured_notation import BeamLevelState as B


class TestWellFormedGroups(unittest.TestCase):
    def test_a_plain_group_is_valid(self) -> None:
        findings = validate_level([B.BEGIN, B.CONTINUE, B.END], level=1)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 1)

    def test_a_two_note_group_is_valid(self) -> None:
        findings = validate_level([B.BEGIN, B.END], level=1)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 1)

    def test_consecutive_groups_are_counted_separately(self) -> None:
        findings = validate_level([B.BEGIN, B.END, B.BEGIN, B.END], level=1)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 2)

    def test_flags_between_groups_are_fine(self) -> None:
        findings = validate_level([B.FLAG, B.BEGIN, B.END, B.FLAG], level=1)

        self.assertTrue(findings.valid)

    def test_levels_that_do_not_apply_are_fine(self) -> None:
        findings = validate_level([B.NOT_APPLICABLE] * 4, level=2)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 0)


class TestImpossibleGroups(unittest.TestCase):
    def test_a_group_that_never_ends_is_reported(self) -> None:
        findings = validate_level([B.BEGIN, B.CONTINUE], level=1)

        self.assertFalse(findings.valid)
        self.assertEqual(findings.unclosed, [0])

    def test_states_inside_a_group_that_was_never_opened_are_all_reported(self) -> None:
        # Both are unopened, not just the first: reporting only the first would
        # understate how much of the sequence is unengravable.
        findings = validate_level([B.CONTINUE, B.END], level=1)

        self.assertEqual(findings.unopened, [0, 1])

    def test_a_second_begin_inside_a_group_is_reported(self) -> None:
        findings = validate_level([B.BEGIN, B.BEGIN, B.END], level=1)

        self.assertEqual(findings.nested, [1])
        self.assertEqual(findings.unclosed, [0])

    def test_a_flag_interrupting_an_open_group_is_reported(self) -> None:
        # An engraver cannot draw a beam that runs through an unbeamed note.
        findings = validate_level([B.BEGIN, B.FLAG, B.END], level=1)

        self.assertEqual(findings.unclosed, [0])
        self.assertEqual(findings.unopened, [2])

    def test_positions_point_at_the_offending_note(self) -> None:
        findings = validate_level([B.FLAG, B.FLAG, B.END], level=1)

        self.assertEqual(findings.unopened, [2])


class TestHooks(unittest.TestCase):
    def test_a_hook_at_a_secondary_level_is_fine(self) -> None:
        findings = validate_level([B.FORWARD_HOOK, B.BACKWARD_HOOK], level=2)

        self.assertTrue(findings.valid)

    def test_a_hook_at_level_one_is_reported(self) -> None:
        # A hook is a fragment of a secondary beam pointing back at a primary one. At
        # level 1 there is no primary beam for it to attach to.
        findings = validate_level([B.BACKWARD_HOOK], level=1)

        self.assertEqual(findings.hook_at_primary_level, [0])

    def test_a_hook_inside_a_group_does_not_close_it(self) -> None:
        findings = validate_level([B.BEGIN, B.BACKWARD_HOOK, B.END], level=2)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 1)


class TestPerVoice(unittest.TestCase):
    def test_a_secondary_break_where_the_primary_continues_is_valid(self) -> None:
        # The whole reason for per-level states: level 2 ending while level 1 carries on
        # is a real engraving choice, not an inconsistency.
        vectors = [
            (B.BEGIN, B.BEGIN),
            (B.CONTINUE, B.END),
            (B.CONTINUE, B.BEGIN),
            (B.END, B.END),
        ]
        findings = validate_voice(vectors, levels=2)

        self.assertTrue(findings.valid)
        self.assertEqual(findings.groups, 3)

    def test_a_short_vector_is_treated_as_not_applicable(self) -> None:
        findings = validate_voice([(B.BEGIN,), (B.END,)], levels=4)

        self.assertTrue(findings.valid)

    def test_findings_from_every_level_are_reported(self) -> None:
        vectors = [(B.BEGIN, B.CONTINUE), (B.END, B.CONTINUE)]
        findings = validate_voice(vectors, levels=2)

        self.assertFalse(findings.valid)
        self.assertTrue(findings.unopened)

    def test_the_description_says_what_went_wrong(self) -> None:
        findings = validate_level([B.BEGIN, B.CONTINUE], level=1)

        self.assertIn("unclosed", findings.describe())


class TestNothingIsRewritten(unittest.TestCase):
    def test_the_input_is_left_exactly_as_given(self) -> None:
        # The raw prediction and any repair must stay distinguishable, or a model that
        # got the beaming right cannot be told from one whose output was corrected.
        states = [B.BEGIN, B.CONTINUE]
        before = list(states)

        validate_level(states, level=1)

        self.assertEqual(states, before)


if __name__ == "__main__":
    unittest.main()
