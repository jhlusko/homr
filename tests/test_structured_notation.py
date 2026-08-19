import unittest

from homr.transformer.structured_notation import (
    DYNAMIC_CLASSES,
    TRAINED_DYNAMIC_MARKS,
    DynamicMark,
    dynamic_mark_from_tag,
    trained_dynamic_mark,
)


class TestDynamicMarkFromTag(unittest.TestCase):
    def test_a_recognised_tag_maps_directly(self) -> None:
        self.assertEqual(dynamic_mark_from_tag("sfz"), DynamicMark.SFZ)

    def test_none_maps_to_none(self) -> None:
        self.assertEqual(dynamic_mark_from_tag(None), DynamicMark.NONE)

    def test_an_unrecognised_tag_maps_to_other(self) -> None:
        self.assertEqual(dynamic_mark_from_tag("madeup"), DynamicMark.OTHER)

    def test_a_hybrid_concatenated_tag_maps_to_other(self) -> None:
        self.assertEqual(dynamic_mark_from_tag("pother-dynamics"), DynamicMark.OTHER)


class TestTrainedDynamicMarks(unittest.TestCase):
    """28.1 (phase16): a head trained on the full ~33-mark representation cannot learn
    marks with single- or low-double-digit support regardless of loss reweighting, so a
    trained head's own vocabulary is a smaller, explicit subset - representation and
    trained-head vocabulary are deliberately decoupled, the same split
    TRAINED_BEAM_LEVELS/TRAINED_SLUR_SLOTS already make for beams and slurs."""

    def test_none_and_other_are_always_trained(self) -> None:
        self.assertIn(DynamicMark.NONE, TRAINED_DYNAMIC_MARKS)
        self.assertIn(DynamicMark.OTHER, TRAINED_DYNAMIC_MARKS)

    def test_a_well_supported_mark_is_kept(self) -> None:
        self.assertEqual(trained_dynamic_mark(DynamicMark.F), DynamicMark.F)

    def test_a_long_tail_mark_folds_to_other(self) -> None:
        self.assertEqual(trained_dynamic_mark(DynamicMark.SFZ), DynamicMark.OTHER)

    def test_a_mark_already_none_stays_none(self) -> None:
        self.assertEqual(trained_dynamic_mark(DynamicMark.NONE), DynamicMark.NONE)

    def test_dynamic_classes_is_the_trained_subset_not_the_full_representation(self) -> None:
        self.assertEqual(set(DYNAMIC_CLASSES), TRAINED_DYNAMIC_MARKS)
        self.assertLess(len(DYNAMIC_CLASSES), len(DynamicMark))

    def test_every_trained_class_maps_to_itself(self) -> None:
        # A collapsed mark that is not its own preimage would mean a trainable class is
        # unreachable - the head could never be supervised to predict it.
        for mark in DYNAMIC_CLASSES:
            self.assertEqual(trained_dynamic_mark(mark), mark)


if __name__ == "__main__":
    unittest.main()
