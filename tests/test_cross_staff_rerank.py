import unittest

from homr.cross_staff_rerank import rerank_staff_candidates
from homr.transformer.vocabulary import EncodedSymbol


def _sym(rhythm: str) -> EncodedSymbol:
    return EncodedSymbol(rhythm)


def _measure(*durations: str) -> list[EncodedSymbol]:
    return [_sym(d) for d in durations] + [_sym("barline")]


class TestRerankStaffCandidates(unittest.TestCase):
    def test_no_alternatives_keeps_the_greedy_decode(self) -> None:
        greedy = _measure("note_4") * 1
        candidates = {0: [greedy], 1: [greedy], 2: [greedy]}

        result = rerank_staff_candidates(candidates)

        self.assertEqual(result, {0: greedy, 1: greedy, 2: greedy})

    def test_an_alternative_matching_the_majority_replaces_a_wrong_greedy_decode(
        self,
    ) -> None:
        # Three corroborating staves land on cumulative position 1 (whole note) at the
        # first barline; the flagged staff's greedy decode landed on 3/4 instead, but
        # one of its forked alternatives matches the majority exactly.
        majority_measure = _measure("note_4", "note_4", "note_4", "note_4")
        wrong_greedy = _measure("note_4", "note_4", "note_4")
        correct_alternative = _measure("note_4", "note_4", "note_4", "note_4")

        candidates = {
            0: [majority_measure],
            1: [majority_measure],
            2: [majority_measure],
            3: [wrong_greedy, correct_alternative],
        }

        result = rerank_staff_candidates(candidates)

        self.assertEqual(result[3], correct_alternative)
        # Untouched staves are returned exactly as given.
        self.assertEqual(result[0], majority_measure)

    def test_an_alternative_that_does_not_improve_agreement_is_not_picked(self) -> None:
        majority_measure = _measure("note_4", "note_4", "note_4", "note_4")
        greedy = _measure("note_4", "note_4", "note_4")
        worse_alternative = _measure("note_4")  # even further from the majority

        candidates = {
            0: [majority_measure],
            1: [majority_measure],
            2: [majority_measure],
            3: [greedy, worse_alternative],
        }

        result = rerank_staff_candidates(candidates)

        self.assertEqual(result[3], greedy)

    def test_fewer_than_min_corroborating_staves_skips_reranking(self) -> None:
        majority_measure = _measure("note_4", "note_4", "note_4", "note_4")
        greedy = _measure("note_4", "note_4", "note_4")
        matching_alternative = _measure("note_4", "note_4", "note_4", "note_4")

        # Only one other staff - below the default min_corroborating_staves=2 bar.
        candidates = {
            0: [majority_measure],
            1: [greedy, matching_alternative],
        }

        result = rerank_staff_candidates(candidates)

        self.assertEqual(result[1], greedy)

    def test_a_staff_with_no_barlines_at_all_is_not_used_as_corroboration(self) -> None:
        no_barlines = [_sym("note_4")]  # never reaches a barline
        majority_measure = _measure("note_4", "note_4", "note_4", "note_4")
        greedy = _measure("note_4", "note_4", "note_4")
        matching_alternative = _measure("note_4", "note_4", "note_4", "note_4")

        candidates = {
            0: [no_barlines],
            1: [majority_measure],
            2: [greedy, matching_alternative],
        }

        # Only one real corroborating staff (index 1) - still below the default bar of 2.
        result = rerank_staff_candidates(candidates)

        self.assertEqual(result[2], greedy)

    def test_a_tie_between_candidates_keeps_the_greedy_default(self) -> None:
        majority_measure = _measure("note_4", "note_4", "note_4", "note_4")
        greedy = _measure("note_4", "note_4", "note_4", "note_4")  # already matches
        equally_good_alternative = _measure("note_4", "note_4", "note_4", "note_4")

        candidates = {
            0: [majority_measure],
            1: [majority_measure],
            2: [greedy, equally_good_alternative],
        }

        result = rerank_staff_candidates(candidates)

        self.assertEqual(result[2], greedy)


if __name__ == "__main__":
    unittest.main()
