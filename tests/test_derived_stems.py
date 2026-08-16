import unittest

from homr.transformer.structured_notation import BeamLevelState as B
from homr.transformer.structured_notation import StemDirection
from training.transformer.derived_stems import Note, derive, groups_from_beams


def _v(*states: B) -> tuple[B, ...]:
    return states


class TestGrouping(unittest.TestCase):
    def test_a_well_formed_run_is_one_group(self) -> None:
        groups = groups_from_beams([_v(B.BEGIN), _v(B.CONTINUE), _v(B.END)])

        self.assertEqual(groups, [[0, 1, 2]])

    def test_a_flag_stands_alone(self) -> None:
        groups = groups_from_beams([_v(B.FLAG), _v(B.BEGIN), _v(B.END)])

        self.assertEqual(groups, [[0], [1, 2]])

    def test_an_unbeamable_note_stands_alone(self) -> None:
        groups = groups_from_beams([_v(B.NOT_APPLICABLE), _v(B.BEGIN), _v(B.END)])

        self.assertEqual(groups, [[0], [1, 2]])

    def test_only_level_one_decides_the_group(self) -> None:
        # Deeper levels subdivide a group that has already chosen its direction.
        groups = groups_from_beams(
            [_v(B.BEGIN, B.BEGIN), _v(B.CONTINUE, B.END), _v(B.END, B.NOT_APPLICABLE)]
        )

        self.assertEqual(groups, [[0, 1, 2]])


class TestMalformedPredictions(unittest.TestCase):
    """A head can emit vectors an engraver never would, and the rule must survive them."""

    def test_a_begin_with_no_end_still_closes(self) -> None:
        groups = groups_from_beams([_v(B.BEGIN), _v(B.CONTINUE), _v(B.BEGIN), _v(B.END)])

        self.assertEqual(groups, [[0, 1], [2, 3]])

    def test_a_continuation_with_nothing_open_stands_alone(self) -> None:
        groups = groups_from_beams([_v(B.CONTINUE), _v(B.BEGIN), _v(B.END)])

        self.assertEqual(groups, [[0], [1, 2]])

    def test_a_flag_inside_a_run_ends_it(self) -> None:
        groups = groups_from_beams([_v(B.BEGIN), _v(B.FLAG), _v(B.END)])

        self.assertEqual(groups, [[0], [1], [2]])

    def test_an_empty_vector_does_not_crash(self) -> None:
        self.assertEqual(groups_from_beams([()]), [[0]])


class TestDerivation(unittest.TestCase):
    def _notes(self, *positions: int) -> list[Note]:
        return [Note(p, StemDirection.UNKNOWN, i) for i, p in enumerate(positions)]

    def test_a_group_takes_one_direction_from_its_furthest_note(self) -> None:
        # Two notes below the middle line and one far above: the group hangs off the
        # extreme, which is what an engraver does, so all three share a down stem.
        notes = self._notes(-1, -2, 6)
        directions = derive(notes, [_v(B.BEGIN), _v(B.CONTINUE), _v(B.END)])

        self.assertEqual(set(directions), {StemDirection.DOWN})

    def test_notes_in_different_groups_may_differ(self) -> None:
        notes = self._notes(-3, 4)
        directions = derive(notes, [_v(B.FLAG), _v(B.FLAG)])

        self.assertEqual(directions, [StemDirection.UP, StemDirection.DOWN])

    def test_a_note_on_the_middle_line_takes_a_down_stem(self) -> None:
        directions = derive(self._notes(0), [_v(B.FLAG)])

        self.assertEqual(directions, [StemDirection.DOWN])

    def test_grouping_changes_the_answer(self) -> None:
        # The whole point: the same pitches beamed together and apart give different
        # directions, which is why a rule with beam grouping beats one without.
        notes = self._notes(-1, 5)
        together = derive(notes, [_v(B.BEGIN), _v(B.END)])
        apart = derive(notes, [_v(B.FLAG), _v(B.FLAG)])

        self.assertEqual(together, [StemDirection.DOWN, StemDirection.DOWN])
        self.assertEqual(apart, [StemDirection.UP, StemDirection.DOWN])


if __name__ == "__main__":
    unittest.main()
