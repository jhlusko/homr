import unittest

from homr.score_profile import (
    SCHEMA_VERSION,
    STRING_QUARTET,
    ScorePart,
    ScoreProfile,
    ScoreProfileSchemaError,
)


class TestScorePart(unittest.TestCase):
    def test_only_stable_id_is_required(self) -> None:
        part = ScorePart.from_dict({"stableId": "violin-1"})

        self.assertEqual(part.stable_id, "violin-1")
        self.assertEqual(part.expected_staff_count, 1)
        self.assertEqual(part.likely_clefs, ())
        self.assertFalse(part.lyrics_expected)

    def test_missing_stable_id_is_refused(self) -> None:
        with self.assertRaises(ScoreProfileSchemaError):
            ScorePart.from_dict({"displayName": "Violin I"})

    def test_round_trips_through_to_dict_and_from_dict(self) -> None:
        part = ScorePart(
            stable_id="viola",
            display_name="Viola",
            instrument_family="strings.viola",
            expected_staff_count=1,
            likely_clefs=("C3", "G2"),
            transposition_semitones=0,
            lyrics_expected=False,
        )

        self.assertEqual(ScorePart.from_dict(part.to_dict()), part)


class TestScoreProfile(unittest.TestCase):
    def test_total_staff_count_sums_across_parts(self) -> None:
        profile = ScoreProfile(
            parts=(
                ScorePart("voice", expected_staff_count=1),
                ScorePart("piano", expected_staff_count=2),
            )
        )

        self.assertEqual(profile.total_staff_count, 3)

    def test_expected_staff_pattern_repeats_a_multi_staff_part_adjacently(self) -> None:
        profile = ScoreProfile(
            parts=(
                ScorePart("voice", expected_staff_count=1),
                ScorePart("piano", expected_staff_count=2),
            )
        )

        self.assertEqual(profile.expected_staff_pattern, ("voice", "piano", "piano"))

    def test_expected_staff_pattern_length_matches_total_staff_count(self) -> None:
        self.assertEqual(
            len(STRING_QUARTET.expected_staff_pattern), STRING_QUARTET.total_staff_count
        )

    def test_part_by_id_finds_a_known_part(self) -> None:
        part = STRING_QUARTET.part_by_id("viola")

        self.assertIsNotNone(part)
        self.assertEqual(part.display_name, "Viola")  # type: ignore[union-attr]

    def test_part_by_id_returns_none_for_an_unknown_part(self) -> None:
        self.assertIsNone(STRING_QUARTET.part_by_id("harpsichord"))

    def test_an_empty_profile_is_valid(self) -> None:
        profile = ScoreProfile()

        self.assertEqual(profile.total_staff_count, 0)
        self.assertEqual(profile.expected_staff_pattern, ())

    def test_round_trips_through_to_dict_and_from_dict(self) -> None:
        self.assertEqual(ScoreProfile.from_dict(STRING_QUARTET.to_dict()), STRING_QUARTET)

    def test_to_dict_stamps_the_schema_version(self) -> None:
        self.assertEqual(STRING_QUARTET.to_dict()["schemaVersion"], SCHEMA_VERSION)

    def test_an_unknown_schema_version_is_refused(self) -> None:
        with self.assertRaises(ScoreProfileSchemaError):
            ScoreProfile.from_dict({"schemaVersion": "homr.score-profile.v2", "parts": []})

    def test_a_missing_schema_version_is_refused_not_defaulted(self) -> None:
        # Silently assuming v1 for a payload with no version at all would read a future,
        # incompatible schema as if it were this one.
        with self.assertRaises(ScoreProfileSchemaError):
            ScoreProfile.from_dict({"parts": []})


class TestStringQuartetExample(unittest.TestCase):
    def test_four_single_staff_parts_in_score_order(self) -> None:
        self.assertEqual(
            [part.stable_id for part in STRING_QUARTET.parts],
            ["violin-1", "violin-2", "viola", "cello"],
        )
        self.assertEqual(STRING_QUARTET.total_staff_count, 4)


if __name__ == "__main__":
    unittest.main()
