import unittest

from training.omr_datasets.split_pairs_by_score import score_of, split_by_score


def _line(score: str, system: int, voice: int = 0) -> str:
    stem = f"{score}-sys{system}-v{voice}"
    return f"/out/{stem}.png,/out/{stem}.tokens"


class TestScoreOf(unittest.TestCase):
    def test_reads_the_score_id_from_the_image_stem(self) -> None:
        self.assertEqual(score_of(_line("IMSLP123", 4)), "IMSLP123")

    def test_score_ids_containing_hyphens_survive(self) -> None:
        self.assertEqual(score_of(_line("IMSLP-odd-9", 2)), "IMSLP-odd-9")

    def test_unparseable_line_returns_none(self) -> None:
        self.assertIsNone(score_of("/out/garbage.png,/out/garbage.tokens"))


class TestSplitByScore(unittest.TestCase):
    def _corpus(self, n_scores: int, per_score: int = 4) -> list[str]:
        return [
            _line(f"IMSLP{i:04d}", s)
            for i in range(n_scores)
            for s in range(per_score)
        ]

    def test_no_score_appears_in_both_splits(self) -> None:
        train, val = split_by_score(self._corpus(60), 0.2)

        self.assertEqual({score_of(x) for x in train} & {score_of(x) for x in val}, set())

    def test_every_line_lands_in_exactly_one_split(self) -> None:
        corpus = self._corpus(40)

        train, val = split_by_score(corpus, 0.25)

        self.assertEqual(len(train) + len(val), len(corpus))
        self.assertEqual(set(train) | set(val), set(corpus))

    def test_all_systems_of_a_score_stay_together(self) -> None:
        corpus = self._corpus(50, per_score=6)

        train, val = split_by_score(corpus, 0.2)

        for split in (train, val):
            counts: dict[str, int] = {}
            for line in split:
                counts[score_of(line)] = counts.get(score_of(line), 0) + 1
            self.assertTrue(all(c == 6 for c in counts.values()))

    def test_split_is_deterministic_across_calls(self) -> None:
        corpus = self._corpus(50)

        first = split_by_score(corpus, 0.2)
        second = split_by_score(corpus, 0.2)

        self.assertEqual(first, second)

    def test_input_order_does_not_change_the_split(self) -> None:
        corpus = self._corpus(50)

        forward = split_by_score(corpus, 0.2)
        backward = split_by_score(list(reversed(corpus)), 0.2)

        self.assertEqual(set(forward[1]), set(backward[1]))

    def test_adding_a_new_score_does_not_move_existing_ones(self) -> None:
        # Stability under growth: recovered pairs get appended later, and that must
        # not reshuffle which scores are held out.
        base = self._corpus(40)
        grown = base + [_line("IMSLP9999", s) for s in range(3)]

        base_val = {score_of(x) for x in split_by_score(base, 0.2)[1]}
        grown_val = {score_of(x) for x in split_by_score(grown, 0.2)[1]}

        self.assertTrue(base_val <= grown_val)

    def test_adding_systems_to_an_existing_score_keeps_it_on_its_side(self) -> None:
        base = self._corpus(40)
        val_score = score_of(split_by_score(base, 0.2)[1][0])
        grown = base + [_line(val_score, 99)]

        train, val = split_by_score(grown, 0.2)

        self.assertIn(val_score, {score_of(x) for x in val})
        self.assertNotIn(val_score, {score_of(x) for x in train})

    def test_a_larger_fraction_holds_out_more(self) -> None:
        corpus = self._corpus(100)

        small = len(split_by_score(corpus, 0.1)[1])
        large = len(split_by_score(corpus, 0.4)[1])

        self.assertLess(small, large)


if __name__ == "__main__":
    unittest.main()
