import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder

try:  # The encoder pulls in timm/torchvision, which a minimal environment may lack.
    from training.architecture.transformer.tromr_arch import TrOMR

    _FULL_STACK = True
except Exception:  # noqa: BLE001
    _FULL_STACK = False


def _config(enable: bool) -> Config:
    config = Config()
    # A small model: this is about wiring and gradient flow, not capacity.
    config.decoder_dim = 32
    config.decoder_depth = 2
    config.decoder_heads = 2
    config.max_seq_len = 16
    config.enable_structured_heads = enable
    config.structured_beam_levels = 2
    config.structured_slur_slots = 1
    return config


def _batch(config: Config, length: int = 6) -> dict:
    def ids(count: int) -> torch.Tensor:
        return torch.randint(0, max(count - 1, 1), (2, length))

    return {
        # The decoder cross-attends to the encoder's output, so a context is required
        # even when the test only cares about the decoder.
        "context": torch.zeros(2, 5, config.decoder_dim),
        "rhythms": ids(config.num_rhythm_tokens),
        "pitchs": ids(config.num_pitch_tokens),
        "lifts": ids(config.num_lift_tokens),
        "articulations": ids(config.num_articulation_tokens),
        "slurs": ids(config.num_slur_tokens),
        "positions": ids(config.num_position_tokens),
        "mask": torch.ones(2, length, dtype=torch.bool),
    }


class TestStructuredHeadsWiring(unittest.TestCase):
    def test_heads_are_absent_unless_enabled(self) -> None:
        # A checkpoint trained before these existed must load into a model that behaves
        # exactly as it did, so the default has to be off.
        self.assertIsNone(get_decoder(_config(enable=False)).structured_heads)
        self.assertIsNotNone(get_decoder(_config(enable=True)).structured_heads)

    def test_existing_losses_are_bit_identical_with_heads_attached(self) -> None:
        # The frozen-core experiment is only answerable if attaching the heads changes
        # nothing about the existing objective.
        torch.manual_seed(0)
        without = get_decoder(_config(enable=False))
        torch.manual_seed(0)
        with_heads = get_decoder(_config(enable=True))

        batch = _batch(_config(enable=False))
        without.eval()
        with_heads.eval()
        with torch.no_grad():
            plain = without(**batch)
            grown = with_heads(**batch)

        for key in ("loss", "loss_rhythm", "loss_pitch", "loss_slurs", "loss_position"):
            self.assertTrue(
                torch.equal(plain[key], grown[key]),
                f"{key} moved when the structured heads were attached",
            )

    def test_structured_logits_appear_only_when_enabled(self) -> None:
        batch = _batch(_config(enable=False))
        with torch.no_grad():
            plain = get_decoder(_config(enable=False))(**batch)
            grown = get_decoder(_config(enable=True))(**batch)

        self.assertIsNone(plain["structured_logits"])
        self.assertEqual(
            sorted(grown["structured_logits"]),
            [
                "advance.delta",
                "beam.level.1",
                "beam.level.2",
                "dynamic.mark",
                "slur.slot.1.event",
                "slur.slot.1.side",
                "stem.direction",
                "tie.state",
            ],
        )

    def test_the_hidden_state_is_exposed_for_the_heads_to_read(self) -> None:
        batch = _batch(_config(enable=False))
        with torch.no_grad():
            result = get_decoder(_config(enable=False))(**batch)

        self.assertEqual(result["hidden"].shape[-1], 32)


@unittest.skipUnless(_FULL_STACK, "needs the full training stack (timm/torchvision)")
class TestFrozenCore(unittest.TestCase):
    def test_only_the_structured_heads_stay_trainable(self) -> None:
        model = TrOMR(_config(enable=True))
        trainable = model.freeze_core_for_structured_heads()

        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("decoder.structured_heads.") for name in trainable))
        frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
        self.assertTrue(any(n.startswith("encoder.") for n in frozen))
        self.assertTrue(any("to_logits_rhythm" in n for n in frozen))

    def test_freezing_without_heads_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            TrOMR(_config(enable=False)).freeze_core_for_structured_heads()


if __name__ == "__main__":
    unittest.main()
