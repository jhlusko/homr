import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.profile_context import (
    ProfileContext,
    ProfileContextEmbedding,
)


def _config() -> Config:
    config = Config()
    # A small model: this is about wiring and gradient flow, not capacity - the same
    # sizing test_structured_heads_wiring.py uses for the same reason.
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
        "positions": ids(config.num_position_tokens),
        "mask": torch.ones(2, length, dtype=torch.bool),
    }


class TestProfileContextWiring(unittest.TestCase):
    def test_omitting_profile_context_emb_is_bit_identical_to_before_this_existed(self) -> None:
        # The same guarantee test_structured_heads_wiring.py makes for the structured
        # heads, made here for score-profile conditioning: a caller that does not know
        # or care about this parameter must see no difference at all.
        torch.manual_seed(0)
        model = get_decoder(_config())
        model.eval()
        batch = _batch(_config())

        with torch.no_grad():
            without_param = model(**batch)
            with_explicit_none = model(**batch, profile_context_emb=None)

        for key in ("loss", "loss_rhythm", "loss_pitch"):
            self.assertTrue(torch.equal(without_param[key], with_explicit_none[key]))

    def test_a_zero_init_profile_context_embedding_changes_nothing_end_to_end(self) -> None:
        # The real integration point, not just ProfileContextEmbedding in isolation:
        # a freshly constructed model plus a freshly constructed (zero-gated)
        # ProfileContextEmbedding, wired together the way real training would, must
        # reproduce the model's own loss exactly.
        torch.manual_seed(0)
        model = get_decoder(_config())
        model.eval()
        profile_module = ProfileContextEmbedding(dim=_config().decoder_dim)
        batch = _batch(_config())
        contexts = [
            ProfileContext(
                instrument_family="strings.violin",
                part_ordinal=0,
                staff_within_part=0,
                expected_staff_count=1,
                likely_clefs=("G2",),
                transposition_semitones=0,
            ),
            None,
        ]

        with torch.no_grad():
            baseline = model(**batch)
            profile_emb = profile_module(contexts)
            conditioned = model(**batch, profile_context_emb=profile_emb)

        for key in ("loss", "loss_rhythm", "loss_pitch"):
            self.assertTrue(torch.equal(baseline[key], conditioned[key]))

    def test_moving_the_gate_actually_changes_the_loss(self) -> None:
        # The inverse of the two tests above: this confirms the zero results are
        # because the gate is zero, not because profile_context_emb is silently
        # ignored somewhere in the wiring.
        torch.manual_seed(0)
        model = get_decoder(_config())
        model.eval()
        profile_module = ProfileContextEmbedding(dim=_config().decoder_dim)
        with torch.no_grad():
            profile_module.gate.fill_(1.0)
        batch = _batch(_config())
        contexts = [
            ProfileContext(
                instrument_family="strings.violin",
                part_ordinal=0,
                staff_within_part=0,
                expected_staff_count=1,
                likely_clefs=("G2",),
                transposition_semitones=0,
            ),
            None,
        ]

        with torch.no_grad():
            baseline = model(**batch)
            profile_emb = profile_module(contexts)
            conditioned = model(**batch, profile_context_emb=profile_emb)

        self.assertFalse(torch.equal(baseline["loss"], conditioned["loss"]))


if __name__ == "__main__":
    unittest.main()
