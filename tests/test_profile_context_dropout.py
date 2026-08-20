import random
import unittest

from training.architecture.transformer.profile_context import (
    ProfileContext,
    apply_context_dropout,
)


def _context(**overrides) -> ProfileContext:
    defaults = dict(
        instrument_family="strings.violin",
        part_ordinal=0,
        staff_within_part=0,
        expected_staff_count=1,
        likely_clefs=("G2",),
        transposition_semitones=0,
    )
    defaults.update(overrides)
    return ProfileContext(**defaults)


class TestApplyContextDropout(unittest.TestCase):
    def test_an_already_missing_context_stays_missing(self) -> None:
        rng = random.Random(0)

        self.assertIsNone(apply_context_dropout(None, rng))

    def test_a_low_roll_drops_to_no_profile(self) -> None:
        rng = random.Random(0)
        rng.random = lambda: 0.1  # type: ignore[method-assign]

        self.assertIsNone(apply_context_dropout(_context(), rng, no_profile_prob=0.3))

    def test_a_mid_roll_masks_instrument_family_only(self) -> None:
        rng = random.Random(0)
        rng.random = lambda: 0.4  # type: ignore[method-assign]

        result = apply_context_dropout(
            _context(), rng, no_profile_prob=0.3, partial_mask_prob=0.3
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.instrument_family, "")
        # Everything else survives - only instrument identity is masked.
        self.assertEqual(result.part_ordinal, 0)
        self.assertEqual(result.likely_clefs, ("G2",))
        self.assertEqual(result.expected_staff_count, 1)

    def test_a_high_roll_leaves_the_context_untouched(self) -> None:
        rng = random.Random(0)
        rng.random = lambda: 0.9  # type: ignore[method-assign]

        result = apply_context_dropout(
            _context(), rng, no_profile_prob=0.3, partial_mask_prob=0.3
        )

        self.assertEqual(result, _context())

    def test_boundary_between_no_profile_and_partial_mask(self) -> None:
        rng = random.Random(0)
        rng.random = lambda: 0.3  # type: ignore[method-assign]

        result = apply_context_dropout(
            _context(), rng, no_profile_prob=0.3, partial_mask_prob=0.3
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.instrument_family, "")

    def test_zero_probabilities_never_alter_the_context(self) -> None:
        rng = random.Random(0)

        for _ in range(50):
            result = apply_context_dropout(
                _context(), rng, no_profile_prob=0.0, partial_mask_prob=0.0
            )
            self.assertEqual(result, _context())

    def test_probability_one_no_profile_always_drops(self) -> None:
        rng = random.Random(0)

        for _ in range(50):
            self.assertIsNone(
                apply_context_dropout(_context(), rng, no_profile_prob=1.0)
            )


if __name__ == "__main__":
    unittest.main()
