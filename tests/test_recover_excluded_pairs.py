import unittest

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.music_xml_parser import Measure
from training.omr_datasets.recover_excluded_pairs import (
    align_near_expected,
    slice_voice_measures,
    window_bounds,
)


def _note(pitch: str) -> EncodedSymbol:
    return EncodedSymbol("note_4", pitch, "_", "_", "_", "upper")


def _clef() -> EncodedSymbol:
    return EncodedSymbol("clef_G2", "_", "_", "_", "_", "upper")


def _stream(measures: list[list[str]]) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    owner: list[int] = []
    for i, m in enumerate(measures):
        for t in m:
            tokens.append(t)
            owner.append(i)
    return tokens, owner


class TestWindowBounds(unittest.TestCase):
    def test_selects_only_tokens_inside_the_measure_range(self) -> None:
        _, owner = _stream([["a"], ["b"], ["c"], ["d"]])

        self.assertEqual(window_bounds(owner, 1, 2), (1, 3))

    def test_empty_slice_when_no_measure_falls_in_range(self) -> None:
        _, owner = _stream([["a"], ["b"]])

        self.assertEqual(window_bounds(owner, 10, 12), (0, 0))

    def test_clamps_naturally_to_what_exists(self) -> None:
        _, owner = _stream([["a"], ["b"], ["c"]])

        self.assertEqual(window_bounds(owner, 0, 99), (0, 3))


class TestAlignNearExpected(unittest.TestCase):
    def test_prefers_the_occurrence_near_the_expected_position(self) -> None:
        # The identical passage ["x","y"] appears at measure 1 and again at
        # measure 9 - a strophic repeat. A global alignment would take the first;
        # windowing around the expected position must take the near one.
        measures = [["a"], ["x", "y"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["x", "y"]]
        tokens, owner = _stream(measures)

        result = align_near_expected(["x", "y"], tokens, owner, expected_start=9, window_measures=2)

        self.assertEqual(result["start_measure"], 9)

    def test_still_finds_the_early_occurrence_when_that_is_the_expected_one(self) -> None:
        measures = [["a"], ["x", "y"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["x", "y"]]
        tokens, owner = _stream(measures)

        result = align_near_expected(["x", "y"], tokens, owner, expected_start=1, window_measures=2)

        self.assertEqual(result["start_measure"], 1)

    def test_returns_none_when_the_window_contains_no_notes(self) -> None:
        tokens, owner = _stream([["a"], ["b"]])

        result = align_near_expected(["a"], tokens, owner, expected_start=50, window_measures=2)

        self.assertIsNone(result)

    def test_a_nearby_repeat_does_not_pull_the_match_backwards(self) -> None:
        # The real failure this ladder exists for: a strophic repeat close enough
        # to sit inside the same window as the true occurrence. Both match
        # perfectly, and difflib prefers the earlier one - so widening in stages
        # must let the nearer candidate win first. Modelled on a real Die Forelle
        # system expected at 45 that recovered to 41.
        measures = [["m%d" % i] for i in range(60)]
        for start in (41, 45):
            measures[start] = ["p", "q"]
            measures[start + 1] = ["r", "s"]
        tokens, owner = _stream(measures)

        result = align_near_expected(
            ["p", "q", "r", "s"], tokens, owner, expected_start=45, window_measures=8
        )

        self.assertEqual(result["start_measure"], 45)

    def test_a_distant_match_is_still_reachable_when_nothing_is_near(self) -> None:
        # Widening must not become a hard proximity constraint: if the only
        # explanation of the crop is further away, it should still be found.
        measures = [["m%d" % i] for i in range(60)]
        measures[38] = ["p", "q"]
        measures[39] = ["r", "s"]
        tokens, owner = _stream(measures)

        result = align_near_expected(
            ["p", "q", "r", "s"], tokens, owner, expected_start=45, window_measures=8
        )

        self.assertEqual(result["start_measure"], 38)

    def test_a_window_miss_is_not_trusted_rather_than_silently_wrong(self) -> None:
        measures = [["a"], ["b"], ["c"], ["d"], ["e"]]
        tokens, owner = _stream(measures)

        result = align_near_expected(
            ["zz", "yy", "xx"], tokens, owner, expected_start=2, window_measures=1
        )

        self.assertTrue(result is None or not result["trusted"])


class TestSliceVoiceMeasures(unittest.TestCase):
    def _voice(self, n: int) -> list:
        voice = []
        for i in range(n):
            measure = Measure([_clef(), _note(f"C{i}")] if i == 0 else [_note(f"C{i}")])
            voice.append(measure)
        return voice

    def test_returns_symbols_for_the_requested_range(self) -> None:
        voice = self._voice(6)

        result = slice_voice_measures(voice, 2, 4)

        pitches = [s.pitch for s in result if s.rhythm.startswith("note")]
        self.assertEqual(pitches, ["C2", "C3"])

    def test_does_not_consume_the_caller_s_voice(self) -> None:
        voice = self._voice(5)

        slice_voice_measures(voice, 1, 3)

        self.assertEqual(len(voice), 5)

    def test_out_of_range_returns_empty(self) -> None:
        voice = self._voice(3)

        self.assertEqual(slice_voice_measures(voice, 1, 99), [])
        self.assertEqual(slice_voice_measures(voice, -1, 2), [])
        self.assertEqual(slice_voice_measures(voice, 2, 2), [])

    def test_a_slice_after_the_start_still_carries_a_clef(self) -> None:
        # The point of walking the cutter forward: a mid-piece slice must still
        # declare the clef in effect, not start bare.
        voice = self._voice(6)

        result = slice_voice_measures(voice, 3, 5)

        self.assertTrue(any(s.rhythm.startswith("clef") for s in result))


if __name__ == "__main__":
    unittest.main()


class TestAlreadyLogged(unittest.TestCase):
    def test_picks_up_recovered_and_failed_scores(self) -> None:
        from training.omr_datasets.recover_excluded_pairs import already_logged

        log = (
            "IMSLP111: recovered 5 pair(s)\n"
            "IMSLP222: FAILED preparing (Octave shift isn't supported)\n"
            "IMSLP333: recovered 0 pair(s)\n"
        )

        self.assertEqual(already_logged(log), {"IMSLP111", "IMSLP222", "IMSLP333"})

    def test_ignores_unrelated_noise(self) -> None:
        from training.omr_datasets.recover_excluded_pairs import already_logged

        log = "Inference Time Tromr: 0.3\nsome warning\n14 recovered, 8 not trusted, 0 skipped\n"

        self.assertEqual(already_logged(log), set())
