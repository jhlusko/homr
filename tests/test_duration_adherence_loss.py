import unittest

import torch

from training.architecture.transformer.decoder import ScoreDecoder

# A tiny, hand-built 4-token rhythm vocabulary, index-matched to a duration/barline
# lookup - deliberately not the real (much larger) vocabulary, so each test case's
# expected numbers are easy to hand-verify. 0=PAD, 1=note_4 (quarter, 1/4 whole note),
# 2=note_8 (eighth, 1/8 whole note), 3=barline (0 duration, closes a measure).
_PAD, _NOTE_4, _NOTE_8, _BARLINE = 0, 1, 2, 3
_DURATIONS = torch.tensor([0.0, 0.25, 0.125, 0.0])
_IS_BARLINE = torch.tensor([0.0, 0.0, 0.0, 1.0])

_LARGE = 50.0  # a logit scale large enough that softmax is ~1.0 on the chosen index


def _one_hot_logits(token_ids: list[int], vocab_size: int = 4) -> torch.Tensor:
    """(1, seq, vocab_size) logits that softmax to (approximately) a one-hot
    distribution on each position's given token id - a stand-in for "the model
    confidently predicts exactly this," so this loss's expected value is easy to
    hand-verify against the ground-truth-only computation."""
    logits = torch.zeros(1, len(token_ids), vocab_size)
    for t, token_id in enumerate(token_ids):
        logits[0, t, token_id] = _LARGE
    return logits


class _FakeDecoder:
    """Only what `calDurationAdherenceLoss` actually reads - avoids constructing a
    real `ScoreTransformerWrapper`/`Config`/checkpoint for what is otherwise a pure
    tensor-math unit test."""

    def __init__(self) -> None:
        self.rhythm_duration = _DURATIONS
        self.rhythm_is_barline = _IS_BARLINE

    calDurationAdherenceLoss = ScoreDecoder.calDurationAdherenceLoss


class TestDurationAdherenceLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = _FakeDecoder()

    def test_a_perfect_prediction_has_approximately_zero_loss(self) -> None:
        # Four quarter notes then a barline: true measure length 4 * 1/4 = 1.0 whole
        # note, and the model confidently predicts exactly that sequence.
        tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([tokens])
        rhythmsp = _one_hot_logits(tokens)
        mask = torch.ones(1, len(tokens))

        loss = self.decoder.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_a_wrong_confident_prediction_produces_the_expected_drift(self) -> None:
        # Ground truth: four quarter notes (true cumulative at the barline = 1.0).
        # The model confidently predicts an eighth note instead of a quarter at the
        # third position - predicted cumulative = 0.25+0.25+0.125+0.25 = 0.875,
        # exactly a 0.125 drift from the true 1.0 at the one barline in this sequence.
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        predicted_tokens = [_NOTE_4, _NOTE_4, _NOTE_8, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits(predicted_tokens)
        mask = torch.ones(1, len(true_tokens))

        loss = self.decoder.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)

        self.assertAlmostEqual(loss.item(), 0.125, places=3)

    def test_padded_positions_do_not_affect_the_loss(self) -> None:
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        # Garbage padding appended after the real content - should be fully masked out.
        padded_true = true_tokens + [_NOTE_8, _NOTE_8]
        padded_predicted = true_tokens + [_NOTE_4, _NOTE_4]
        rhythmso = torch.tensor([padded_true])
        rhythmsp = _one_hot_logits(padded_predicted)
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])

        loss = self.decoder.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_a_sequence_with_no_barlines_has_zero_loss(self) -> None:
        # Nothing to check adherence against - the denominator clamp (not a division
        # by zero) is what this test guards.
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4]
        wrong_tokens = [_NOTE_8, _NOTE_8, _NOTE_8]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits(wrong_tokens)
        mask = torch.ones(1, len(true_tokens))

        loss = self.decoder.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)

        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_the_loss_is_differentiable_with_respect_to_the_logits(self) -> None:
        true_tokens = [_NOTE_4, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE]
        rhythmso = torch.tensor([true_tokens])
        rhythmsp = _one_hot_logits([_NOTE_8, _NOTE_4, _NOTE_4, _NOTE_4, _BARLINE])
        rhythmsp.requires_grad_(True)
        mask = torch.ones(1, len(true_tokens))

        loss = self.decoder.calDurationAdherenceLoss(rhythmsp, rhythmso, mask)
        loss.backward()

        self.assertIsNotNone(rhythmsp.grad)
        self.assertGreater(float(rhythmsp.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
