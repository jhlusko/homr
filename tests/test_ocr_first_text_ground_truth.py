import unittest

from training.omr_datasets.ocr_first_text_ground_truth import (
    match_dynamics_to_ocr,
    match_lyrics_to_ocr,
    match_verses_to_ocr,
    page_measure_ranges,
)


def _ocr_line(text: str) -> dict:
    return {"box": {"left": 0, "top": 0, "width": 10, "height": 10}, "text": text, "score": 0.9}


class TestPageMeasureRanges(unittest.TestCase):
    def test_computes_cumulative_ranges(self) -> None:
        pages = [[3, 4], [5, 3, 4], [4]]

        ranges = page_measure_ranges(pages)

        self.assertEqual(ranges, [(0, 7), (7, 19), (19, 23)])

    def test_empty_pages_list(self) -> None:
        self.assertEqual(page_measure_ranges([]), [])

    def test_a_page_with_no_systems_is_a_zero_width_range(self) -> None:
        self.assertEqual(page_measure_ranges([[3], []]), [(0, 3), (3, 3)])


class TestMatchLyricsToOcr(unittest.TestCase):
    def test_confirms_a_line_whose_words_mostly_match(self) -> None:
        expected = ["Fried", "li", "cher", "Abend"]
        lines = [_ocr_line("Fried li cher Abend")]

        matches = match_lyrics_to_ocr(expected, lines)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["kind"], "lyric")

    def test_rejects_a_line_that_does_not_match(self) -> None:
        expected = ["Fried", "li", "cher"]
        lines = [_ocr_line("Allegro moderato")]

        matches = match_lyrics_to_ocr(expected, lines)

        self.assertEqual(matches, [])

    def test_tolerates_some_ocr_error(self) -> None:
        # One garbled token ("cber" for "cher") among four real words still clears
        # the default 0.6 line-match threshold.
        expected = ["Fried", "li", "cher", "Abend"]
        lines = [_ocr_line("Fried li cber Abend")]

        matches = match_lyrics_to_ocr(expected, lines)

        self.assertEqual(len(matches), 1)

    def test_ignores_lines_with_no_tokens(self) -> None:
        matches = match_lyrics_to_ocr(["word"], [_ocr_line("   ")])

        self.assertEqual(matches, [])

    def test_strips_punctuation_before_comparing(self) -> None:
        expected = ["Abend"]
        lines = [_ocr_line("Abend.")]

        matches = match_lyrics_to_ocr(expected, lines)

        self.assertEqual(len(matches), 1)


class TestMatchDynamicsToOcr(unittest.TestCase):
    def test_matches_a_standalone_dynamic_mark(self) -> None:
        matches = match_dynamics_to_ocr(["p", "f", "cresc"], [_ocr_line("f")])

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["kind"], "dynamic")

    def test_does_not_match_an_unrelated_line(self) -> None:
        matches = match_dynamics_to_ocr(["p", "f"], [_ocr_line("Allegro moderato")])

        self.assertEqual(matches, [])

    def test_does_not_match_a_multi_word_line_against_a_short_mark(self) -> None:
        # A whole-line comparison against a one-letter mark should not fire just
        # because the mark happens to be a substring of a much longer line.
        matches = match_dynamics_to_ocr(["f"], [_ocr_line("far away from here")])

        self.assertEqual(matches, [])


def _ocr_line_at(text: str, top: int) -> dict:
    return {"box": {"left": 0, "top": top, "width": 10, "height": 10}, "text": text, "score": 0.9}


class TestMatchVersesToOcr(unittest.TestCase):
    def test_each_verse_matches_its_own_printed_line(self) -> None:
        words_per_verse = {"1": ["Fried", "li", "cher"], "2": ["An", "de", "re"]}
        lines = [_ocr_line_at("Fried li cher", 0), _ocr_line_at("An de re", 20)]

        matches = match_verses_to_ocr(words_per_verse, lines)

        self.assertEqual(len(matches), 2)
        by_verse = {m["verse"]: m["text"] for m in matches}
        self.assertEqual(by_verse["1"], "Fried li cher")
        self.assertEqual(by_verse["2"], "An de re")

    def test_a_line_already_claimed_by_one_verse_is_not_reused_by_another(self) -> None:
        # Both verses happen to share every word - without exclusion, the second
        # verse would double-claim the first verse's own line.
        words_per_verse = {"1": ["gleich"], "2": ["gleich"]}
        lines = [_ocr_line_at("gleich", 0)]

        matches = match_verses_to_ocr(words_per_verse, lines)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["verse"], "1")

    def test_verses_are_tried_in_numeric_order(self) -> None:
        words_per_verse = {"2": ["Zweite"], "1": ["Erste"]}
        lines = [_ocr_line_at("Erste", 0), _ocr_line_at("Zweite", 20)]

        matches = match_verses_to_ocr(words_per_verse, lines)

        self.assertEqual([m["verse"] for m in matches], ["1", "2"])

    def test_no_matching_line_for_a_verse_is_not_an_error(self) -> None:
        words_per_verse = {"1": ["Fried", "li", "cher"]}
        lines = [_ocr_line_at("completely unrelated text", 0)]

        matches = match_verses_to_ocr(words_per_verse, lines)

        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
