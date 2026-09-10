import unittest

import torch

from training.architecture.transformer.decoder import ScoreDecoder

# Same tiny 5-token vocabulary test_duration_adherence_loss.py uses.
_PAD, _NOTE_4, _NOTE_8, _BARLINE, _CHORD = 0, 1, 2, 3, 4
_DURATIONS = torch.tensor([0.0, 0.25, 0.125, 0.0, 0.0])
_IS_BARLINE = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
_IS_CHORD_MARKER = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])

_LARGE = 50.0


def _one_hot_logits(token_ids: list[int], vocab_size: int = 5) -> torch.Tensor:
    logits = torch.zeros(1, len(token_ids), vocab_size)
    for t, token_id in enumerate(token_ids):
        logits[0, t, token_id] = _LARGE
    return logits


class _FakeDecoder:
    """Only what `calCrossStaffCoherenceLoss` actually reads."""

    def __init__(self) -> None:
        self.rhythm_duration = _DURATIONS
        self.rhythm_is_barline = _IS_BARLINE
        self.rhythm_is_chord_marker = _IS_CHORD_MARKER

    calCrossStaffCoherenceLoss = ScoreDecoder.calCrossStaffCoherenceLoss
    _not_chord_continuation = ScoreDecoder._not_chord_continuation


class TestCrossStaffCoherenceLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = _FakeDecoder()

    def test_matching_the_systems_ground_truth_has_approximately_zero_loss(self) -> None:
        # Four quarter notes then a barline: this staff predicts exactly that, and its
        # system's own curve (from siblings) says the first measure should sum to 1.0.
        tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([tokens])
        rhythmsp = _one_hot_logits(tokens)
        mask = torch.ones(1, len(tokens))
        curve = torch.tensor([[1.0] + [0.0] * 31])
        curve_len = torch.tensor([1])
        present = torch.tensor([1.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_diverging_from_the_systems_ground_truth_produces_the_expected_drift(
        self,
    ) -> None:
        # This staff's own label also says four quarter notes (1.0), but it *predicts*
        # an eighth note at the third position - predicted cumulative 0.875, a 0.125
        # drift from the system's shared target of 1.0 (not from this staff's own
        # label, which is the point of this loss vs. calDurationAdherenceLoss).
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        predicted_tokens = [_NOTE_4, _NOTE_4, _NOTE_8, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits(predicted_tokens)
        mask = torch.ones(1, len(true_tokens))
        curve = torch.tensor([[1.0] + [0.0] * 31])
        curve_len = torch.tensor([1])
        present = torch.tensor([1.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )

        self.assertAlmostEqual(loss.item(), 0.125, places=3)

    def test_a_two_note_chord_is_not_double_counted(self) -> None:
        # Same chord shape test_duration_adherence_loss.py's own regression test
        # uses: note_4, chord, note_4, barline sums to one quarter (0.25), not two.
        tokens = [_NOTE_4, _CHORD, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([tokens])
        rhythmsp = _one_hot_logits(tokens)
        mask = torch.ones(1, len(tokens))
        curve = torch.tensor([[0.25] + [0.0] * 31])
        curve_len = torch.tensor([1])
        present = torch.tensor([1.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_a_sample_with_no_system_curve_contributes_nothing(self) -> None:
        # present=0.0: an unresolvable system (no fragment, non-OSSQ, etc.) - this
        # sample's own drift must not be silently treated as if the target were 0.0.
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        predicted_tokens = [_NOTE_8, _NOTE_8, _NOTE_8, _NOTE_8, _BARLINE]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits(predicted_tokens)
        mask = torch.ones(1, len(true_tokens))
        curve = torch.zeros(1, 32)
        curve_len = torch.tensor([0])
        present = torch.tensor([0.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_a_barline_beyond_the_curves_own_length_is_not_compared(self) -> None:
        # The system curve only covers 1 measure (this system happened to be short),
        # but this staff's own decode reaches a second barline - that second barline
        # is out of the curve's range and must not silently compare against padding.
        tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE] * 2
        rhythmso = torch.tensor([tokens])
        # Second measure predicted wildly wrong - should not move the loss at all,
        # since only the first (in-range) barline is compared.
        wrong_second_measure = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE, *([_NOTE_8] * 4), _BARLINE]
        rhythmsp = _one_hot_logits(wrong_second_measure)
        mask = torch.ones(1, len(tokens))
        curve = torch.tensor([[1.0] + [0.0] * 31])
        curve_len = torch.tensor([1])
        present = torch.tensor([1.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_the_loss_is_differentiable_with_respect_to_the_logits(self) -> None:
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits([_NOTE_8, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE])
        rhythmsp.requires_grad_(True)
        mask = torch.ones(1, len(true_tokens))
        curve = torch.tensor([[1.0] + [0.0] * 31])
        curve_len = torch.tensor([1])
        present = torch.tensor([1.0])

        loss = self.decoder.calCrossStaffCoherenceLoss(
            rhythmsp, rhythmso, mask, curve, curve_len, present
        )
        loss.backward()

        self.assertIsNotNone(rhythmsp.grad)
        self.assertGreater(float(rhythmsp.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
