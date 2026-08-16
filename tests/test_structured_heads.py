import unittest

import torch
from torch import nn

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    SLUR_EVENT_CLASSES,
    SLUR_SIDE_CLASSES,
    STEM_CLASSES,
    BeamLevelState,
    StemDirection,
)
from training.architecture.transformer.checkpoint_loading import (
    CheckpointMismatch,
    load_checkpoint,
)
from training.architecture.transformer.structured_heads import (
    StructuredNotationHeads,
    head_names,
)


class TestHeadNames(unittest.TestCase):
    def test_names_are_the_review_contract_identifiers(self) -> None:
        names = head_names(beam_levels=2, slur_slots=1)

        self.assertEqual(
            names,
            [
                "beam.level.1",
                "beam.level.2",
                "stem.direction",
                "slur.slot.1.event",
                "slur.slot.1.side",
            ],
        )

    def test_default_configuration_follows_the_support_tables(self) -> None:
        # Four beam levels, not six: levels 5 and 6 have no training examples at all.
        # Two slur slots, not six: slots 3-6 hold 116 training occurrences between them.
        names = head_names()

        self.assertIn("beam.level.4", names)
        self.assertNotIn("beam.level.5", names)
        self.assertIn("slur.slot.2.event", names)
        self.assertNotIn("slur.slot.3.event", names)


class TestStructuredNotationHeads(unittest.TestCase):
    def setUp(self) -> None:
        self.heads = StructuredNotationHeads(dim=16, beam_levels=2, slur_slots=1)
        self.hidden = torch.zeros(3, 7, 16)

    def test_one_logits_tensor_per_head(self) -> None:
        logits = self.heads(self.hidden)

        self.assertEqual(sorted(logits), sorted(self.heads.head_names()))

    def test_shapes_keep_batch_and_sequence(self) -> None:
        logits = self.heads(self.hidden)

        self.assertEqual(logits["beam.level.1"].shape, (3, 7, len(BEAM_LEVEL_CLASSES)))
        self.assertEqual(logits["stem.direction"].shape, (3, 7, len(STEM_CLASSES)))
        self.assertEqual(logits["slur.slot.1.event"].shape, (3, 7, len(SLUR_EVENT_CLASSES)))
        self.assertEqual(logits["slur.slot.1.side"].shape, (3, 7, len(SLUR_SIDE_CLASSES)))

    def test_every_beam_state_is_predictable(self) -> None:
        # FLAG and NOT_APPLICABLE are ordinary predictions, not sentinels, so the head
        # must have a class for each.
        self.assertEqual(len(BEAM_LEVEL_CLASSES), len(BeamLevelState))

    def test_the_stem_head_cannot_predict_unknown(self) -> None:
        # UNKNOWN marks a source that does not say. It is masked out of the loss, and a
        # head able to emit it could learn silence as an answer.
        self.assertNotIn(StemDirection.UNKNOWN, STEM_CLASSES)
        self.assertIn(StemDirection.DOUBLE, STEM_CLASSES)

    def test_a_zero_head_configuration_is_still_valid(self) -> None:
        heads = StructuredNotationHeads(dim=8, beam_levels=0, slur_slots=0)

        self.assertEqual(list(heads(torch.zeros(1, 2, 8))), ["stem.direction"])

    def test_negative_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StructuredNotationHeads(dim=8, beam_levels=-1)


class _Core(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 4)


class _Grown(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.structured_heads = nn.Linear(4, 3)


class TestCheckpointLoading(unittest.TestCase):
    def test_new_heads_may_be_absent_and_are_reported(self) -> None:
        old = _Core().state_dict()

        report = load_checkpoint(_Grown(), old, expected_new_prefixes=["structured_heads"])

        self.assertEqual(
            sorted(report.initialized), ["structured_heads.bias", "structured_heads.weight"]
        )

    def test_anything_else_missing_is_an_error(self) -> None:
        # The failure strict=False hides: a renamed or absent core layer loads silently
        # and the model trains from a worse starting point than the checkpoint it named.
        old = _Core().state_dict()
        del old["body.weight"]

        with self.assertRaises(CheckpointMismatch) as ctx:
            load_checkpoint(_Grown(), old, expected_new_prefixes=["structured_heads"])

        self.assertIn("body.weight", str(ctx.exception))

    def test_a_checkpoint_key_with_no_home_is_an_error(self) -> None:
        state = _Grown().state_dict()
        state["stray.weight"] = torch.zeros(2)

        with self.assertRaises(CheckpointMismatch) as ctx:
            load_checkpoint(_Grown(), state, expected_new_prefixes=["structured_heads"])

        self.assertIn("stray.weight", str(ctx.exception))

    def test_an_exact_checkpoint_initializes_nothing(self) -> None:
        report = load_checkpoint(_Grown(), _Grown().state_dict(), ["structured_heads"])

        self.assertEqual(report.initialized, ())
        self.assertIn("nothing left to initialize", report.describe())

    def test_no_allowlist_means_no_missing_parameters_allowed(self) -> None:
        with self.assertRaises(CheckpointMismatch):
            load_checkpoint(_Grown(), _Core().state_dict())


if __name__ == "__main__":
    unittest.main()
