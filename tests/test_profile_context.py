import unittest

import torch

from homr.score_profile import ScorePart
from training.architecture.transformer.profile_context import (
    MAX_CLEF_SLOTS,
    ProfileContext,
    ProfileContextEmbedding,
    context_to_batch_fields,
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


class TestZeroInitBackwardCompatibility(unittest.TestCase):
    """§7.2's own stated requirement: the unconditioned path must be bit-identical at
    initialization. This is the guarantee that makes it safe to land this module ahead
    of any training data actually using it - a fresh model with this module attached
    and never given real context behaves exactly like one without it.
    """

    def test_a_real_context_contributes_exactly_zero_at_init(self) -> None:
        module = ProfileContextEmbedding(dim=16)

        vector = module.embed_one(_context(), device=torch.device("cpu"))

        self.assertTrue(torch.equal(vector, torch.zeros(16)))

    def test_a_missing_context_also_contributes_exactly_zero_at_init(self) -> None:
        module = ProfileContextEmbedding(dim=16)

        vector = module.embed_one(None, device=torch.device("cpu"))

        self.assertTrue(torch.equal(vector, torch.zeros(16)))

    def test_a_batch_of_mixed_contexts_is_all_zero_at_init(self) -> None:
        module = ProfileContextEmbedding(dim=8)

        batch = module.forward([_context(), None, _context(part_ordinal=3)])

        self.assertEqual(batch.shape, (3, 8))
        self.assertTrue(torch.equal(batch, torch.zeros(3, 8)))

    def test_after_the_gate_moves_a_real_context_is_no_longer_zero(self) -> None:
        # Confirms the zero result above is because the gate is zero, not because the
        # sub-embeddings themselves happen to be zero - moving the gate away from zero
        # (as training would) must actually change the output.
        module = ProfileContextEmbedding(dim=16)
        with torch.no_grad():
            module.gate.fill_(1.0)

        vector = module.embed_one(_context(), device=torch.device("cpu"))

        self.assertFalse(torch.equal(vector, torch.zeros(16)))

    def test_after_the_gate_moves_missing_and_present_differ(self) -> None:
        module = ProfileContextEmbedding(dim=16)
        with torch.no_grad():
            module.gate.fill_(1.0)

        present = module.embed_one(_context(), device=torch.device("cpu"))
        missing = module.embed_one(None, device=torch.device("cpu"))

        self.assertFalse(torch.equal(present, missing))


class TestBucketing(unittest.TestCase):
    def test_an_unrecognised_instrument_family_does_not_crash(self) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        vector = module.embed_one(
            _context(instrument_family="some.unrecognised.family"), device=torch.device("cpu")
        )

        self.assertEqual(vector.shape, (8,))

    def test_an_out_of_range_part_ordinal_is_clipped_not_a_crash(self) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        vector = module.embed_one(_context(part_ordinal=999), device=torch.device("cpu"))

        self.assertEqual(vector.shape, (8,))

    def test_an_out_of_range_transposition_is_clamped_not_a_crash(self) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        vector = module.embed_one(
            _context(transposition_semitones=999), device=torch.device("cpu")
        )

        self.assertEqual(vector.shape, (8,))

    def test_an_empty_clef_set_does_not_crash(self) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        vector = module.embed_one(_context(likely_clefs=()), device=torch.device("cpu"))

        self.assertEqual(vector.shape, (8,))

    def test_two_different_clef_sets_produce_different_vectors(self) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        violin = module.embed_one(_context(likely_clefs=("G2",)), device=torch.device("cpu"))
        cello = module.embed_one(
            _context(likely_clefs=("F4", "C4", "G2")), device=torch.device("cpu")
        )

        self.assertFalse(torch.equal(violin, cello))


def _stack_batch_fields(*field_dicts: dict) -> dict:
    """Mimics what HuggingFace Trainer's default collator does: stack each key across
    samples into a batched tensor."""
    keys = field_dicts[0].keys()
    return {
        key: torch.tensor([fields[key] for fields in field_dicts], dtype=torch.long)
        for key in keys
    }


class TestContextToBatchFields(unittest.TestCase):
    def test_a_missing_context_is_present_zero(self) -> None:
        fields = context_to_batch_fields(None)

        self.assertEqual(fields["profile_present"], 0)
        self.assertEqual(fields["profile_clef_indices"], [0] * MAX_CLEF_SLOTS)
        self.assertEqual(fields["profile_clef_count"], 0)

    def test_a_real_context_is_present_one(self) -> None:
        fields = context_to_batch_fields(_context())

        self.assertEqual(fields["profile_present"], 1)

    def test_clef_indices_are_padded_to_the_fixed_slot_count(self) -> None:
        fields = context_to_batch_fields(_context(likely_clefs=("G2",)))

        self.assertEqual(len(fields["profile_clef_indices"]), MAX_CLEF_SLOTS)
        self.assertEqual(fields["profile_clef_count"], 1)

    def test_more_clefs_than_slots_are_capped_not_an_error(self) -> None:
        fields = context_to_batch_fields(
            _context(likely_clefs=("G2", "F4", "C3", "C4", "C5"))
        )

        self.assertEqual(len(fields["profile_clef_indices"]), MAX_CLEF_SLOTS)
        self.assertEqual(fields["profile_clef_count"], MAX_CLEF_SLOTS)


class TestBatchAndListAgree(unittest.TestCase):
    """`forward_from_batch` (the training-facing, vectorized entry point) and
    `forward`/`embed_one` (the direct-caller entry point) must compute the same vector
    for the same logical context - the property `context_to_batch_fields`'s own
    docstring names as the reason it exists.
    """

    def _assert_batch_matches_list(self, contexts: list) -> None:
        module = ProfileContextEmbedding(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)

        from_list = module.forward(contexts)
        batch_fields = _stack_batch_fields(*[context_to_batch_fields(c) for c in contexts])
        from_batch = module.forward_from_batch(**batch_fields)

        self.assertTrue(
            torch.allclose(from_list, from_batch, atol=1e-6),
            f"forward vs forward_from_batch diverged:\n{from_list}\nvs\n{from_batch}",
        )

    def test_a_full_context_matches(self) -> None:
        self._assert_batch_matches_list([_context()])

    def test_a_missing_context_matches(self) -> None:
        self._assert_batch_matches_list([None])

    def test_an_empty_clef_set_matches(self) -> None:
        self._assert_batch_matches_list([_context(likely_clefs=())])

    def test_a_single_clef_matches(self) -> None:
        self._assert_batch_matches_list([_context(likely_clefs=("G2",))])

    def test_a_full_three_clef_set_matches(self) -> None:
        self._assert_batch_matches_list([_context(likely_clefs=("F4", "C4", "G2"))])

    def test_a_mixed_batch_matches(self) -> None:
        self._assert_batch_matches_list(
            [
                _context(instrument_family="strings.cello", likely_clefs=("F4", "C4", "G2")),
                None,
                _context(part_ordinal=3, transposition_semitones=-2, likely_clefs=()),
            ]
        )


class TestFromScorePart(unittest.TestCase):
    def test_carries_the_score_parts_own_fields(self) -> None:
        part = ScorePart(
            "viola", instrument_family="strings.viola", likely_clefs=("C3", "G2")
        )

        context = ProfileContext.from_score_part(part, part_ordinal=2)

        self.assertEqual(context.instrument_family, "strings.viola")
        self.assertEqual(context.part_ordinal, 2)
        self.assertEqual(context.staff_within_part, 0)
        self.assertEqual(context.likely_clefs, ("C3", "G2"))

    def test_staff_within_part_defaults_to_zero(self) -> None:
        part = ScorePart("piano", expected_staff_count=2)

        context = ProfileContext.from_score_part(part, part_ordinal=0)

        self.assertEqual(context.staff_within_part, 0)

    def test_staff_within_part_can_be_supplied_explicitly(self) -> None:
        part = ScorePart("piano", expected_staff_count=2)

        context = ProfileContext.from_score_part(part, part_ordinal=0, staff_within_part=1)

        self.assertEqual(context.staff_within_part, 1)


if __name__ == "__main__":
    unittest.main()
