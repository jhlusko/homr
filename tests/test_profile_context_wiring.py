import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder
from training.architecture.transformer.profile_context import (
    ProfileContext,
    context_to_batch_fields,
)

try:  # The encoder pulls in timm/torchvision, which a minimal environment may lack.
    from training.architecture.transformer.tromr_arch import TrOMR

    _FULL_STACK = True
except Exception:  # noqa: BLE001
    _FULL_STACK = False


def _config(enable_profile_context: bool = False) -> Config:
    config = Config()
    # A small model: this is about wiring and gradient flow, not capacity - the same
    # sizing test_structured_heads_wiring.py uses for the same reason.
    config.decoder_dim = 32
    config.decoder_depth = 2
    config.decoder_heads = 2
    config.max_seq_len = 16
    config.enable_profile_context = enable_profile_context
    return config


def _profile_batch_fields(*contexts: "ProfileContext | None") -> dict:
    """Mimics what HuggingFace Trainer's default collator does to a batch of
    DataLoader.__getitem__ samples: stack each key across samples into one tensor.
    profile_clef_indices is already a per-sample tensor, so it stacks via torch.stack;
    every other field is a plain int.
    """
    per_sample = [context_to_batch_fields(context) for context in contexts]
    result = {}
    for key in per_sample[0]:
        values = [sample[key] for sample in per_sample]
        result[key] = (
            torch.stack(values) if key == "profile_clef_indices" else torch.tensor(values)
        )
    return result


def _batch(config: Config, length: int = 6, with_profile_context: bool = False) -> dict:
    def ids(count: int) -> torch.Tensor:
        return torch.randint(0, max(count - 1, 1), (2, length))

    batch = {
        "context": torch.zeros(2, 5, config.decoder_dim),
        "rhythms": ids(config.num_rhythm_tokens),
        "pitchs": ids(config.num_pitch_tokens),
        "lifts": ids(config.num_lift_tokens),
        "articulations": ids(config.num_articulation_tokens),
        "slurs": ids(config.num_slur_tokens),
        "positions": ids(config.num_position_tokens),
        "mask": torch.ones(2, length, dtype=torch.bool),
    }
    if with_profile_context:
        context = ProfileContext(
            instrument_family="strings.violin",
            part_ordinal=0,
            staff_within_part=0,
            expected_staff_count=1,
            likely_clefs=("G2",),
            transposition_semitones=0,
        )
        batch.update(_profile_batch_fields(context, None))
    return batch


class TestProfileContextWiring(unittest.TestCase):
    def test_disabled_by_default_is_bit_identical_to_before_this_existed(self) -> None:
        # A checkpoint trained before this existed must load into a model that behaves
        # exactly as it did - the same guarantee test_structured_heads_wiring.py makes
        # for the structured heads.
        torch.manual_seed(0)
        without = get_decoder(_config(enable_profile_context=False))
        torch.manual_seed(0)
        with_module = get_decoder(_config(enable_profile_context=True))
        without.eval()
        with_module.eval()
        batch = _batch(_config())

        with torch.no_grad():
            plain = without(**batch)
            grown = with_module(**batch)

        for key in ("loss", "loss_rhythm", "loss_pitch"):
            self.assertTrue(torch.equal(plain[key], grown[key]))

    def test_enabled_but_zero_gated_changes_nothing_even_with_real_batch_fields(self) -> None:
        # The real integration point: profile_* index tensors actually present in the
        # batch, exactly as DataLoader.__getitem__ + the default collator would supply
        # them, routed through ScoreDecoder's own profile_context submodule - the gate
        # being zero at initialization must still make this a no-op.
        torch.manual_seed(0)
        model = get_decoder(_config(enable_profile_context=True))
        model.eval()
        batch_without = _batch(_config(), with_profile_context=False)
        batch_with = _batch(_config(), with_profile_context=True)
        # Same rhythm/pitch/... tensors either way - only the profile_* fields differ.
        batch_with.update({k: v for k, v in batch_without.items() if k in batch_with})

        with torch.no_grad():
            baseline = model(**batch_without)
            conditioned = model(**batch_with)

        for key in ("loss", "loss_rhythm", "loss_pitch"):
            self.assertTrue(torch.equal(baseline[key], conditioned[key]))

    def test_moving_the_gate_actually_changes_the_loss(self) -> None:
        # The inverse of the test above: confirms the zero result is because the gate
        # is zero, not because the batch fields are silently dropped somewhere in the
        # popping/threading.
        torch.manual_seed(0)
        model = get_decoder(_config(enable_profile_context=True))
        model.eval()
        with torch.no_grad():
            model.profile_context.gate.fill_(1.0)
        batch_without = _batch(_config(), with_profile_context=False)
        batch_with = _batch(_config(), with_profile_context=True)
        batch_with.update({k: v for k, v in batch_without.items() if k in batch_with})

        with torch.no_grad():
            baseline = model(**batch_without)
            conditioned = model(**batch_with)

        self.assertFalse(torch.equal(baseline["loss"], conditioned["loss"]))

    def test_disabled_module_ignores_profile_fields_in_the_batch_without_crashing(self) -> None:
        # A batch that happens to carry profile_* keys (mixed-corpus training, or a
        # dataloader upgraded ahead of the config) must not break a model that has
        # enable_profile_context=False - self.profile_context is None, so
        # _pop_profile_context_emb must still discard the keys cleanly.
        torch.manual_seed(0)
        model = get_decoder(_config(enable_profile_context=False))
        model.eval()
        batch = _batch(_config(), with_profile_context=True)

        with torch.no_grad():
            result = model(**batch)

        self.assertIn("loss", result)


@unittest.skipUnless(_FULL_STACK, "needs the full training stack (timm/torchvision)")
class TestFreezeCoreForProfileContext(unittest.TestCase):
    def test_only_profile_context_stays_trainable(self) -> None:
        model = TrOMR(_config(enable_profile_context=True))

        trainable = model.freeze_core_for_profile_context()

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("decoder.profile_context.") for name in trainable))
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.assertTrue(any(n.startswith("encoder.") for n in frozen))
        self.assertTrue(any("to_logits_rhythm" in n for n in frozen))

    def test_freezing_without_the_module_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            TrOMR(_config(enable_profile_context=False)).freeze_core_for_profile_context()


if __name__ == "__main__":
    unittest.main()
