import contextlib
import io
import unittest

import torch
from torch import nn

from training.transformer.train_staff_context import (
    evaluate,
    flatten_staff_dim,
    masked_mean_pool,
    mixed_first_pass_hidden,
    set_probe_mode,
    staff_context_parameters,
    train_epoch,
    two_pass_forward,
)


class _FakeStaffContext(nn.Module):
    """Same shape as the real `StaffContextTransformer`: a `(staff_hidden, mask) ->
    context` module with a gate a test can move off zero."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, staff_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.gate * self.linear(staff_hidden) * mask.unsqueeze(-1).float()


class _FakeNet(nn.Module):
    """Stand-in for `ScoreTransformerWrapper`: shares one `nn.Embedding` +
    `core` Linear so `_FakeModel.forward` (still the old teacher-forced-only path)
    and `mixed_first_pass_hidden` (which calls this directly, exactly like the real
    `model.decoder.net`) compute the same thing given the same rhythms. `context` is
    accepted, unused - the real net's `x` shape follows the query (rhythms) length,
    not context's, and this fake only needs to be shape-consistent, not semantically
    faithful to cross-attention.
    """

    def __init__(self, embed: nn.Embedding, core: nn.Linear, to_vocab: nn.Linear) -> None:
        super().__init__()
        self.embed = embed
        self.core = core
        self.to_vocab = to_vocab

    def forward(
        self,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: object = None,
        return_center_of_attention: bool = False,
        **kwargs: object,
    ) -> tuple:
        x = self.core(self.embed(rhythms))
        logits = self.to_vocab(x)
        return (logits, logits, logits, logits, logits, logits, x, None, None)


class _FakeEncoder(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


class _FakeDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.staff_context = _FakeStaffContext()
        self.core = nn.Linear(4, 4)
        self.embed = nn.Embedding(5, 4)
        self.to_vocab = nn.Linear(4, 5)
        self.net = _FakeNet(self.embed, self.core, self.to_vocab)


class _FakeModel(nn.Module):
    """A stand-in with the shape `two_pass_forward` depends on: a `"hidden"` output
    trimmed the same way the real decoder trims `mask` to `rhythms[:, :-1]`'s length,
    a `staff_context_emb` kwarg that additively biases it before the loss, and an
    `encoder`/`decoder.net` pair `mixed_first_pass_hidden` calls directly - built from
    the same `embed`/`core` the old teacher-forced-only forward below uses, so
    `test_zero_gate_makes_the_two_passes_identical` still holds regardless of how the
    (now genuinely different) first pass computes its pooled hidden state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _FakeDecoder()
        self.encoder = _FakeEncoder()

    def forward(
        self,
        inputs: torch.Tensor,
        rhythms: torch.Tensor,
        pitchs: torch.Tensor,
        lifts: torch.Tensor,
        articulations: torch.Tensor,
        slurs: torch.Tensor,
        mask: torch.Tensor,
        sampling_prob: float = 1.0,
        staff_context_emb: torch.Tensor | None = None,
    ) -> dict:
        hidden = self.decoder.core(self.decoder.embed(rhythms))[:, :-1]
        if staff_context_emb is not None:
            hidden = hidden + staff_context_emb.unsqueeze(1)
        loss = (hidden**2).mean()
        return {"hidden": hidden, "loss": loss}


def _batch(staff_count: int = 3, real_staves: int = 2, seq_len: int = 5) -> dict:
    staff_mask = torch.tensor(
        [[True] * real_staves + [False] * (staff_count - real_staves)]
    )
    token_field = torch.zeros(1, staff_count, seq_len, dtype=torch.long)
    return {
        "inputs": torch.randn(1, staff_count, seq_len, 4),
        "rhythms": token_field.clone(),
        "pitchs": token_field.clone(),
        "lifts": token_field.clone(),
        "articulations": token_field.clone(),
        "slurs": token_field.clone(),
        "mask": torch.ones(1, staff_count, seq_len, dtype=torch.bool),
        "staff_mask": staff_mask,
    }


class TestFlattenStaffDim(unittest.TestCase):
    def test_stacks_batch_and_staff_into_one_leading_dimension(self) -> None:
        batch = _batch(staff_count=3)

        flat, sample_count, staff_count = flatten_staff_dim(batch)

        self.assertEqual(sample_count, 1)
        self.assertEqual(staff_count, 3)
        self.assertEqual(flat["inputs"].shape[0], 3)
        self.assertNotIn("staff_mask", flat)


class TestMaskedMeanPool(unittest.TestCase):
    def test_pools_only_over_real_tokens(self) -> None:
        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[True, True, False]])

        pooled = masked_mean_pool(hidden, mask)

        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 2.0]])))

    def test_an_all_masked_row_does_not_divide_by_zero(self) -> None:
        hidden = torch.randn(1, 3, 2)
        mask = torch.zeros(1, 3, dtype=torch.bool)

        pooled = masked_mean_pool(hidden, mask)

        self.assertTrue(torch.isfinite(pooled).all())


class TestSetProbeMode(unittest.TestCase):
    def test_the_core_is_in_eval_mode(self) -> None:
        model = _FakeModel()
        model.train()

        set_probe_mode(model)

        self.assertFalse(model.decoder.core.training)

    def test_staff_context_is_in_train_mode(self) -> None:
        model = _FakeModel()
        model.eval()

        set_probe_mode(model)

        self.assertTrue(model.decoder.staff_context.training)

    def test_a_model_with_no_staff_context_does_not_crash(self) -> None:
        class _NoStaffContextModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = nn.Linear(1, 1)

        set_probe_mode(_NoStaffContextModel())  # must not raise


class TestStaffContextParameters(unittest.TestCase):
    def test_only_staff_context_parameters_are_returned(self) -> None:
        model = _FakeModel()

        params = staff_context_parameters(model)
        names = [
            name
            for name, param in model.named_parameters()
            if any(param is p for p in params)
        ]

        self.assertTrue(names)
        self.assertTrue(all(name.startswith("decoder.staff_context.") for name in names))


class TestTwoPassForward(unittest.TestCase):
    def test_returns_both_passes_with_a_loss_each(self) -> None:
        model = _FakeModel()
        model.eval()

        outputs = two_pass_forward(model, _batch(), device="cpu")

        self.assertIn("loss", outputs["first_pass"])
        self.assertIn("loss", outputs["second_pass"])

    def test_zero_gate_makes_the_two_passes_identical(self) -> None:
        # StaffContextTransformer's own zero-init gate means the second pass must be
        # bit-identical to the first until training moves the gate - the same
        # guarantee test_staff_context.py checks on the real module in isolation.
        model = _FakeModel()
        model.eval()

        outputs = two_pass_forward(model, _batch(), device="cpu")

        self.assertTrue(
            torch.equal(outputs["first_pass"]["loss"], outputs["second_pass"]["loss"])
        )

    def test_moving_the_gate_changes_the_second_pass(self) -> None:
        model = _FakeModel()
        model.eval()
        with torch.no_grad():
            model.decoder.staff_context.gate.fill_(1.0)

        outputs = two_pass_forward(model, _batch(), device="cpu")

        self.assertFalse(
            torch.equal(outputs["first_pass"]["loss"], outputs["second_pass"]["loss"])
        )


class TestMixedFirstPassHidden(unittest.TestCase):
    def test_sampling_prob_one_is_fully_teacher_forced_deterministically(self) -> None:
        # rand() > 1.0 is never true, so mix_mask is always 0 regardless of the draw -
        # two calls with different global RNG state must still agree exactly.
        model = _FakeModel()
        model.eval()
        flat, _, _ = flatten_staff_dim(_batch())

        torch.manual_seed(0)
        first = mixed_first_pass_hidden(model, flat, sampling_prob=1.0)
        torch.manual_seed(123)
        second = mixed_first_pass_hidden(model, flat, sampling_prob=1.0)

        self.assertTrue(torch.equal(first, second))

    def test_lower_sampling_prob_changes_the_pooled_hidden_state(self) -> None:
        # This is the actual exposure-bias fix: sampling_prob < 1.0 must substitute
        # the model's own greedy prediction into the first pass's input somewhere,
        # not silently reproduce the fully teacher-forced result.
        model = _FakeModel()
        model.eval()
        # Force the fake net's greedy prediction off of token 0 (ground truth's own
        # value in every `_batch()` position) deterministically, so the test does not
        # depend on a random init happening to already predict 0 by chance.
        with torch.no_grad():
            model.decoder.to_vocab.bias.zero_()
            model.decoder.to_vocab.bias[1] = 10.0
        flat, _, _ = flatten_staff_dim(_batch())

        fully_teacher_forced = mixed_first_pass_hidden(model, flat, sampling_prob=1.0)
        fully_substituted = mixed_first_pass_hidden(model, flat, sampling_prob=0.0)

        self.assertFalse(torch.allclose(fully_teacher_forced, fully_substituted))

    def test_does_not_mutate_the_caller_s_batch(self) -> None:
        model = _FakeModel()
        model.eval()
        flat, _, _ = flatten_staff_dim(_batch())
        original_rhythms = flat["rhythms"].clone()

        mixed_first_pass_hidden(model, flat, sampling_prob=0.0)

        self.assertTrue(torch.equal(flat["rhythms"], original_rhythms))


class TestTrainEpoch(unittest.TestCase):
    def test_only_staff_context_weights_move(self) -> None:
        model = _FakeModel()
        with torch.no_grad():
            model.decoder.staff_context.gate.fill_(1.0)
        before_core = model.decoder.core.weight.clone()
        before_staff_context = model.decoder.staff_context.linear.weight.clone()
        for param in model.decoder.core.parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(staff_context_parameters(model), lr=0.1)

        train_epoch(model, [_batch(), _batch()], optimizer, epoch=1, device="cpu")

        self.assertTrue(torch.equal(model.decoder.core.weight, before_core))
        self.assertFalse(
            torch.equal(model.decoder.staff_context.linear.weight, before_staff_context)
        )

    def test_returns_a_report_with_the_mean_loss(self) -> None:
        model = _FakeModel()
        with torch.no_grad():
            model.decoder.staff_context.gate.fill_(1.0)
        optimizer = torch.optim.SGD(staff_context_parameters(model), lr=0.01)

        report = train_epoch(model, [_batch(), _batch()], optimizer, epoch=3, device="cpu")

        self.assertEqual(report["epoch"], 3)
        self.assertEqual(report["batches"], 2)
        self.assertIn("loss", report)

    def test_progress_every_prints_a_mid_epoch_line_with_the_batch_total(self) -> None:
        model = _FakeModel()
        optimizer = torch.optim.SGD(staff_context_parameters(model), lr=0.01)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            train_epoch(
                model, [_batch(), _batch()], optimizer, epoch=1, device="cpu",
                progress_every=1,
            )

        lines = [line for line in out.getvalue().splitlines() if "batch 1" in line]
        self.assertTrue(lines, out.getvalue())
        self.assertIn("/2", lines[0])

    def test_progress_every_zero_disables_mid_epoch_printing(self) -> None:
        model = _FakeModel()
        optimizer = torch.optim.SGD(staff_context_parameters(model), lr=0.01)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            train_epoch(
                model, [_batch(), _batch()], optimizer, epoch=1, device="cpu",
                progress_every=0,
            )

        self.assertNotIn("batch 1", out.getvalue())


class TestEvaluate(unittest.TestCase):
    def test_zero_gate_makes_with_and_without_equal(self) -> None:
        model = _FakeModel()
        model.eval()

        with_context, without_context = evaluate(model, [_batch()], device="cpu")

        self.assertEqual(with_context, without_context)

    def test_moving_the_gate_changes_only_the_with_context_loss(self) -> None:
        model = _FakeModel()
        model.eval()
        with torch.no_grad():
            model.decoder.staff_context.gate.fill_(1.0)

        with_context, without_context = evaluate(model, [_batch()], device="cpu")

        self.assertNotEqual(with_context, without_context)


if __name__ == "__main__":
    unittest.main()
