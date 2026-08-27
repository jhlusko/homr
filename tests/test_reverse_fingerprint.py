import unittest

from training.omr_datasets.reverse_fingerprint import (
    assign_measures_to_systems,
    span_score,
    unclaimed_measures,
)


def build(measures: list[list[str]]) -> tuple[list[str], list[int]]:
    flat, owner = [], []
    for index, measure in enumerate(measures):
        for token in measure:
            flat.append(token)
            owner.append(index)
    return flat, owner


MEASURES = [
    ["C4", "D4"],
    ["E4", "F4"],
    ["G4", "A4"],
    ["B4", "C5"],
    ["D5", "E5"],
    ["F5", "G5"],
]


class TestSpanScore(unittest.TestCase):
    def test_an_exact_span_scores_one(self) -> None:
        gt, owner = build(MEASURES)
        self.assertEqual(span_score(["E4", "F4"], gt, 1, 2, owner), 1.0)

    def test_a_span_much_longer_than_the_crop_is_penalised(self) -> None:
        gt, owner = build(MEASURES)
        tight = span_score(["E4", "F4"], gt, 1, 2, owner)
        loose = span_score(["E4", "F4"], gt, 1, 5, owner)
        self.assertGreater(tight, loose)

    def test_an_empty_span_is_never_a_free_win(self) -> None:
        gt, owner = build(MEASURES)
        self.assertEqual(span_score(["E4"], gt, 2, 2, owner), 0.0)


class TestAssignment(unittest.TestCase):
    def test_systems_are_segmented_in_order_without_overlap(self) -> None:
        gt, owner = build(MEASURES)
        systems = [["C4", "D4", "E4", "F4"], ["G4", "A4", "B4", "C5"], ["D5", "E5", "F5", "G5"]]
        got = assign_measures_to_systems(systems, gt, owner, len(MEASURES))
        self.assertEqual(
            [(a.system, a.start_measure, a.end_measure) for a in got],
            [(0, 0, 2), (1, 2, 4), (2, 4, 6)],
        )

    def test_an_unreadable_system_is_pinned_by_its_neighbours(self) -> None:
        """The whole point of running it in reverse: system 1's reading is garbage,
        but systems 0 and 2 claim the measures either side, so what is left over is
        system 1's span - recovered without its own reading proving anything."""
        gt, owner = build(MEASURES)
        systems = [["C4", "D4", "E4", "F4"], ["zz", "zz"], ["D5", "E5", "F5", "G5"]]
        got = assign_measures_to_systems(systems, gt, owner, len(MEASURES))
        middle = next(a for a in got if a.system == 1)
        self.assertEqual((middle.start_measure, middle.end_measure), (2, 4))

    def test_every_labelled_measure_is_claimed(self) -> None:
        gt, owner = build(MEASURES)
        systems = [["C4", "D4", "E4", "F4"], ["G4", "A4", "B4", "C5"], ["D5", "E5", "F5", "G5"]]
        got = assign_measures_to_systems(systems, gt, owner, len(MEASURES))
        self.assertEqual(unclaimed_measures(got, len(MEASURES)), [])

    def test_a_non_music_detection_can_be_given_nothing(self) -> None:
        gt, owner = build(MEASURES)
        systems = [["C4", "D4", "E4", "F4"], [], ["G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]]
        got = assign_measures_to_systems(systems, gt, owner, len(MEASURES))
        empty = next(a for a in got if a.system == 1)
        self.assertEqual(empty.start_measure, empty.end_measure)
        self.assertEqual(unclaimed_measures(got, len(MEASURES)), [])


if __name__ == "__main__":
    unittest.main()


class TestUnreadableCrops(unittest.TestCase):
    """A crop that yielded no tokens is an abstention, not an empty system.  Scoring
    it zero made the empty move always win, so a rest-heavy system was handed nothing
    - human review called 30 of 30 such systems real music."""

    def test_an_unreadable_crop_still_scores_above_an_empty_span(self) -> None:
        from training.omr_datasets.reverse_fingerprint import (
            EMPTY_SPAN_SCORE,
            UNREADABLE_SPAN_SCORE,
        )
        gt, owner = build(MEASURES)
        self.assertGreater(span_score([], gt, 1, 3, owner), EMPTY_SPAN_SCORE)
        self.assertEqual(span_score([], gt, 1, 3, owner), UNREADABLE_SPAN_SCORE)

    def test_an_unreadable_system_is_placed_not_emptied(self) -> None:
        gt, owner = build(MEASURES)
        systems = [["C4", "D4", "E4", "F4"], [], ["D5", "E5", "F5", "G5"]]
        got = assign_measures_to_systems(systems, gt, owner, len(MEASURES))
        middle = next(a for a in got if a.system == 1)
        self.assertGreater(middle.end_measure, middle.start_measure)
        self.assertEqual((middle.start_measure, middle.end_measure), (2, 4))

    def test_an_empty_gt_window_still_scores_zero(self) -> None:
        gt, owner = build(MEASURES)
        self.assertEqual(span_score([], gt, 2, 2, owner), 0.0)
