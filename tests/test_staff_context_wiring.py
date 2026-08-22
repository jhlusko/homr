import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_score_wrapper


def _config() -> Config:
    # A small model: this is about wiring and gradient flow, not capacity - the same
    # sizing test_profile_context_wiring.py/test_structured_heads_wiring.py use for
    # the same reason.
    config = Config()
    config.decoder_dim = 32
    config.decoder_depth = 2
    config.decoder_heads = 2
    config.max_seq_len = 16
    return config


def _batch(config: Config, length: int = 6) -> dict:
    def ids(count: int) -> torch.Tensor:
        return torch.randint(0, max(count - 1, 1), (2, length))

    return {
        "context": torch.zeros(2, 5, config.decoder_dim),
        "rhythms": ids(config.num_rhythm_tokens),
        "pitchs": ids(config.num_pitch_tokens),
        "lifts": ids(config.num_lift_tokens),
        "articulations": ids(config.num_articulation_tokens),
        "slurs": ids(config.num_slur_tokens),
        "mask": torch.ones(2, length, dtype=torch.bool),
    }


class TestStaffContextEmbWiring(unittest.TestCase):
    def test_omitting_it_is_unaffected(self) -> None:
        config = _config()
        torch.manual_seed(0)
        net = get_score_wrapper(config)
        net.eval()
        batch = _batch(config)

        with torch.no_grad():
            without_kwarg = net(**batch)[0]
            with_none = net(**batch, staff_context_emb=None)[0]

        self.assertTrue(torch.equal(without_kwarg, with_none))

    def test_a_zero_bias_leaves_the_output_unchanged(self) -> None:
        config = _config()
        torch.manual_seed(0)
        net = get_score_wrapper(config)
        net.eval()
        batch = _batch(config)
        zero_bias = torch.zeros(2, config.decoder_dim)

        with torch.no_grad():
            baseline = net(**batch)[0]
            with_zero = net(**batch, staff_context_emb=zero_bias)[0]

        self.assertTrue(torch.allclose(baseline, with_zero, atol=1e-6))

    def test_a_nonzero_bias_changes_the_output(self) -> None:
        config = _config()
        torch.manual_seed(0)
        net = get_score_wrapper(config)
        net.eval()
        batch = _batch(config)
        torch.manual_seed(1)
        bias = torch.randn(2, config.decoder_dim)

        with torch.no_grad():
            baseline = net(**batch)[0]
            biased = net(**batch, staff_context_emb=bias)[0]

        self.assertFalse(torch.allclose(baseline, biased, atol=1e-4))

    def test_independent_of_profile_context_emb(self) -> None:
        """Both biases can be supplied together and each still moves the output -
        neither is silently overwritten by the other, since they are added, not
        assigned."""
        config = _config()
        torch.manual_seed(0)
        net = get_score_wrapper(config)
        net.eval()
        batch = _batch(config)
        torch.manual_seed(1)
        profile_bias = torch.randn(2, config.decoder_dim)
        staff_bias = torch.randn(2, config.decoder_dim)

        with torch.no_grad():
            baseline = net(**batch)[0]
            with_profile_only = net(**batch, profile_context_emb=profile_bias)[0]
            with_both = net(**batch, profile_context_emb=profile_bias, staff_context_emb=staff_bias)[0]

        self.assertFalse(torch.allclose(baseline, with_profile_only, atol=1e-4))
        self.assertFalse(torch.allclose(with_profile_only, with_both, atol=1e-4))

    def test_the_bias_is_differentiable(self) -> None:
        config = _config()
        torch.manual_seed(0)
        net = get_score_wrapper(config)
        batch = _batch(config)
        bias = torch.randn(2, config.decoder_dim, requires_grad=True)

        out = net(**batch, staff_context_emb=bias)[0]
        out.sum().backward()

        self.assertIsNotNone(bias.grad)
        self.assertTrue(torch.isfinite(bias.grad).all())


if __name__ == "__main__":
    unittest.main()
