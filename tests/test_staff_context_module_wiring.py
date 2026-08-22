import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder

try:  # The encoder pulls in timm/torchvision, which a minimal environment may lack.
    from training.architecture.transformer.tromr_arch import TrOMR

    _FULL_STACK = True
except Exception:  # noqa: BLE001
    _FULL_STACK = False


def _config(enable_staff_context: bool = False) -> Config:
    config = Config()
    # A small model: this is about wiring, not capacity - the same sizing
    # test_profile_context_wiring.py/test_structured_heads_wiring.py use for the
    # same reason.
    config.decoder_dim = 32
    config.decoder_depth = 2
    config.decoder_heads = 2
    config.max_seq_len = 16
    config.enable_staff_context = enable_staff_context
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


class TestStaffContextModuleAttachment(unittest.TestCase):
    def test_disabled_by_default_leaves_the_attribute_none(self) -> None:
        decoder = get_decoder(_config(enable_staff_context=False))
        self.assertIsNone(decoder.staff_context)

    def test_enabled_attaches_the_module(self) -> None:
        decoder = get_decoder(_config(enable_staff_context=True))
        self.assertIsNotNone(decoder.staff_context)

    def test_disabled_by_default_is_bit_identical_to_before_this_existed(self) -> None:
        # A checkpoint trained before this existed must load into a model that
        # behaves exactly as it did - the module isn't even in the forward path
        # unless a caller explicitly passes staff_context_emb.
        torch.manual_seed(0)
        without = get_decoder(_config(enable_staff_context=False))
        torch.manual_seed(0)
        with_module = get_decoder(_config(enable_staff_context=True))
        without.eval()
        with_module.eval()
        batch = _batch(_config())

        with torch.no_grad():
            plain = without(**batch)
            grown = with_module(**batch)

        for key in ("loss", "loss_rhythm", "loss_pitch"):
            self.assertTrue(torch.equal(plain[key], grown[key]))


@unittest.skipUnless(_FULL_STACK, "needs the full training stack (timm/torchvision)")
class TestFreezeCoreForStaffContext(unittest.TestCase):
    def test_only_staff_context_stays_trainable(self) -> None:
        model = TrOMR(_config(enable_staff_context=True))

        trainable = model.freeze_core_for_staff_context()

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("decoder.staff_context.") for name in trainable))
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.assertTrue(any(n.startswith("encoder.") for n in frozen))
        self.assertTrue(any("to_logits_rhythm" in n for n in frozen))

    def test_freezing_without_the_module_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            TrOMR(_config(enable_staff_context=False)).freeze_core_for_staff_context()


if __name__ == "__main__":
    unittest.main()
