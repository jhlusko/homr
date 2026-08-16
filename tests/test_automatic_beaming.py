import unittest

from homr.transformer.automatic_beaming import (
    BeamableNote,
    agreement,
    automatic_beams,
    beat_divisions,
)
from homr.transformer.structured_notation import BeamLevelState as B

DIV = 4  # divisions per quarter


def _eighths(count: int, flags: int = 1) -> list[BeamableNote]:
    return [
        BeamableNote(onset=i * (DIV // 2), duration=DIV // 2, flags=flags) for i in range(count)
    ]


def _level(vectors: list, level: int = 1) -> list:
    return [vector[level - 1] for vector in vectors]


class TestBeatLength(unittest.TestCase):
    def test_simple_metres_beat_on_the_denominator(self) -> None:
        self.assertEqual(beat_divisions(4, 4, DIV), DIV)
        self.assertEqual(beat_divisions(3, 4, DIV), DIV)

    def test_compound_metres_beat_in_threes(self) -> None:
        # 6/8 beams by the dotted quarter, not the eighth.
        self.assertEqual(beat_divisions(6, 8, DIV), DIV * 3 // 2 * 1 or DIV * 3 // 2)
        self.assertEqual(beat_divisions(9, 8, DIV), beat_divisions(6, 8, DIV))

    def test_three_eight_is_simple_not_compound(self) -> None:
        # One group of three, not a compound beat, so it beats on the eighth.
        self.assertEqual(beat_divisions(3, 8, DIV), DIV // 2)


class TestGrouping(unittest.TestCase):
    def test_eighths_within_a_beat_are_beamed(self) -> None:
        vectors = automatic_beams(_eighths(2), beat=DIV)

        self.assertEqual(_level(vectors), [B.BEGIN, B.END])

    def test_beams_do_not_cross_a_beat(self) -> None:
        # Four eighths in 4/4 are two groups of two, not one group of four.
        vectors = automatic_beams(_eighths(4), beat=DIV)

        self.assertEqual(_level(vectors), [B.BEGIN, B.END, B.BEGIN, B.END])

    def test_a_compound_beat_holds_all_three(self) -> None:
        vectors = automatic_beams(_eighths(3), beat=DIV * 3 // 2)

        self.assertEqual(_level(vectors), [B.BEGIN, B.CONTINUE, B.END])

    def test_a_rest_ends_a_group(self) -> None:
        notes = [
            BeamableNote(0, 2, 1),
            BeamableNote(2, 2, 1, is_rest=True),
            BeamableNote(4, 2, 1),
            BeamableNote(6, 2, 1),
        ]
        vectors = automatic_beams(notes, beat=DIV)

        self.assertEqual(_level(vectors)[0], B.FLAG)
        self.assertEqual(_level(vectors)[2:], [B.BEGIN, B.END])

    def test_a_quarter_note_ends_a_group_and_carries_nothing(self) -> None:
        notes = [BeamableNote(0, 2, 1), BeamableNote(2, 4, 0), BeamableNote(6, 2, 1)]
        vectors = automatic_beams(notes, beat=DIV)

        self.assertEqual(_level(vectors)[1], B.NOT_APPLICABLE)

    def test_a_lone_eighth_is_a_flag_not_a_one_note_beam(self) -> None:
        # Calling this a beam would count a non-decision as a decision.
        vectors = automatic_beams([BeamableNote(0, 2, 1)], beat=DIV)

        self.assertEqual(_level(vectors), [B.FLAG])


class TestSecondaryLevels(unittest.TestCase):
    def test_sixteenths_beam_at_both_levels(self) -> None:
        notes = [BeamableNote(i, 1, 2) for i in range(4)]
        vectors = automatic_beams(notes, beat=DIV)

        self.assertEqual(_level(vectors, 1), [B.BEGIN, B.CONTINUE, B.CONTINUE, B.END])
        self.assertEqual(_level(vectors, 2), [B.BEGIN, B.CONTINUE, B.CONTINUE, B.END])

    def test_a_mixed_group_breaks_the_secondary_beam_only(self) -> None:
        # An eighth among sixteenths carries the primary beam but not the secondary, so
        # level 2 breaks either side of it while level 1 runs through.
        notes = [
            BeamableNote(0, 1, 2),
            BeamableNote(1, 1, 2),
            BeamableNote(2, 2, 1),
        ]
        vectors = automatic_beams(notes, beat=DIV)

        self.assertEqual(_level(vectors, 1), [B.BEGIN, B.CONTINUE, B.END])
        self.assertEqual(_level(vectors, 2)[:2], [B.BEGIN, B.END])
        self.assertEqual(_level(vectors, 2)[2], B.NOT_APPLICABLE)


class TestAgreement(unittest.TestCase):
    def test_notes_neither_side_beams_are_not_counted(self) -> None:
        # Otherwise the overwhelming majority of a score - quarters and longer - would
        # inflate agreement towards 100% for free.
        plain = [(B.NOT_APPLICABLE,) * 6] * 10
        matching, comparable = agreement(plain, plain)

        self.assertEqual((matching, comparable), (0, 0))

    def test_identical_vectors_agree(self) -> None:
        vectors = automatic_beams(_eighths(4), beat=DIV)

        self.assertEqual(agreement(vectors, vectors), (4, 4))

    def test_a_differing_level_counts_as_a_whole_note_mismatch(self) -> None:
        left = automatic_beams(_eighths(4), beat=DIV)
        right = automatic_beams(_eighths(4), beat=DIV * 2)

        matching, comparable = agreement(left, right)

        self.assertEqual(comparable, 4)
        self.assertLess(matching, 4)


if __name__ == "__main__":
    unittest.main()
