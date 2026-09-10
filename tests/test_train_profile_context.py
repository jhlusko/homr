import unittest

import torch
from torch import nn

from training.transformer.train_profile_context import (
    evaluate,
    profile_context_parameters,
    set_probe_mode,
    train_epoch,
)


class _ProfileContext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(4, 4)

    def forward(self, present: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear(x)) * present.unsqueeze(-1)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.profile_context = _ProfileContext()
        self.core = nn.Linear(4, 1)


class _Model(nn.Module):
    """A stand-in with the shape train_epoch/evaluate depend on: a frozen body, the
    profile_context module, and a forward returning a scalar "loss"."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder()

    def forward(self, inputs: torch.Tensor, profile_present: torch.Tensor) -> dict:
        conditioned = self.decoder.profile_context(profile_present.float(), inputs)
        prediction = self.decoder.core(conditioned)
        loss = (prediction**2).mean()
        return {"loss": loss}


def _batch(present: bool = True) -> dict:
    return {
        "inputs": torch.randn(2, 4),
        "profile_present": torch.full((2,), 1 if present else 0, dtype=torch.long),
    }


class TestSetProbeMode(unittest.TestCase):
    def test_the_core_is_in_eval_mode(self) -> None:
        model = _Model()
        model.train()

        set_probe_mode(model)

        self.assertFalse(model.decoder.core.training)

    def test_profile_context_is_in_train_mode(self) -> None:
        model = _Model()
        model.eval()

        set_probe_mode(model)

        self.assertTrue(model.decoder.profile_context.training)

    def test_a_model_with_no_profile_context_does_not_crash(self) -> None:
        class _NoProfileModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = nn.Linear(1, 1)

        set_probe_mode(_NoProfileModel())  # must not raise


class TestProfileContextParameters(unittest.TestCase):
    def test_only_profile_context_parameters_are_returned(self) -> None:
        model = _Model()

        params = profile_context_parameters(model)
        names = [
            name
            for name, param in model.named_parameters()
            if any(param is p for p in params)
        ]

        self.assertTrue(names)
        self.assertTrue(all(name.startswith("decoder.profile_context.") for name in names))


class TestTrainEpoch(unittest.TestCase):
    def test_only_profile_context_weights_move(self) -> None:
        model = _Model()
        before_core = model.decoder.core.weight.clone()
        before_profile = model.decoder.profile_context.linear.weight.clone()
        for param in model.decoder.core.parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(profile_context_parameters(model), lr=0.1)

        train_epoch(model, [_batch(), _batch()], optimizer, epoch=1)

        self.assertTrue(torch.equal(model.decoder.core.weight, before_core))
        self.assertFalse(torch.equal(model.decoder.profile_context.linear.weight, before_profile))

    def test_returns_a_report_with_the_mean_loss(self) -> None:
        model = _Model()
        optimizer = torch.optim.SGD(profile_context_parameters(model), lr=0.01)

        report = train_epoch(model, [_batch(), _batch()], optimizer, epoch=3)

        self.assertEqual(report["epoch"], 3)
        self.assertEqual(report["batches"], 2)
        self.assertIn("loss", report)


class TestEvaluate(unittest.TestCase):
    def test_force_no_profile_zeroes_the_present_flag(self) -> None:
        model = _Model()
        model.eval()
        batch = _batch(present=True)

        with_profile = evaluate(model, [batch], force_no_profile=False)
        without_profile = evaluate(model, [batch], force_no_profile=True)

        # profile_present gates the whole conditioned path to zero in the stand-in, so
        # forcing it off must change the loss (the stand-in's core sees an all-zero
        # input instead of the real one) rather than silently being ignored.
        self.assertNotEqual(with_profile, without_profile)

    def test_does_not_mutate_the_original_batch(self) -> None:
        model = _Model()
        model.eval()
        batch = _batch(present=True)
        original = batch["profile_present"].clone()

        evaluate(model, [batch], force_no_profile=True)

        self.assertTrue(torch.equal(batch["profile_present"], original))

    def test_eval_mode_is_deterministic_despite_dropout(self) -> None:
        model = _Model()
        batch = _batch()

        first = evaluate(model, [batch])
        second = evaluate(model, [batch])

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
