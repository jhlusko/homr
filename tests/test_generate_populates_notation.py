"""`generate()` must attach the heads' predictions to the symbols it returns.

This is the seam that was missing: the heads trained to 0.9508 exact beam vector and
`build_beams` could write `<beam>`, but `generate()` discarded the hidden state and
`EncodedSymbol.notation` stayed None, so every note came out unbeamed in production.

These tests run the real decoder rather than a stand-in, because the bug they guard
against lives precisely in the wiring between real components - a mock returning a shape
I chose would have passed while production stayed broken.
"""

import unittest

import torch

from homr.transformer.configs import Config
from training.architecture.transformer.decoder import get_decoder


def _config(enable: bool) -> Config:
    """The production decoder shape, with only the sequence length capped.

    Shrinking the model is not an option here, unlike the other structured-head tests:
    `init_cache` hardcodes `(1, 8, cache_len, 64)` over `range(32)` and never consults
    the config, so a decoder with different dim or head count fails inside the attention
    stack before reaching anything this file is about. Capping `max_seq_len` keeps the
    run short without changing any shape.
    """
    config = Config()
    config.max_seq_len = 4
    config.enable_structured_heads = enable
    return config


def _generate(config: Config) -> list:
    decoder = get_decoder(config)
    decoder.eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    nonote = torch.zeros((1, 1), dtype=torch.long)
    with torch.no_grad():
        return decoder.generate(
            start, nonote, context=torch.randn(1, 4, config.decoder_dim)
        )


class TestGeneratePopulatesNotation(unittest.TestCase):
    def test_symbols_carry_notation_when_heads_are_enabled(self) -> None:
        symbols = _generate(_config(enable=True))

        self.assertTrue(symbols, "generate produced no symbols")
        self.assertTrue(
            all(symbol.notation is not None for symbol in symbols),
            "a symbol came back without notation - build_beams will skip it",
        )

    def test_notation_is_none_when_heads_are_disabled(self) -> None:
        # The standing rule in configs.py: a checkpoint trained without the heads must
        # behave exactly as it did before.
        symbols = _generate(_config(enable=False))

        self.assertTrue(symbols)
        self.assertTrue(all(symbol.notation is None for symbol in symbols))

    def test_beam_levels_match_the_configured_count(self) -> None:
        config = _config(enable=True)
        symbols = _generate(config)

        for symbol in symbols:
            self.assertEqual(
                len(symbol.notation.beam_levels), config.structured_beam_levels
            )

    def test_slur_slots_match_the_configured_count(self) -> None:
        config = _config(enable=True)
        symbols = _generate(config)

        for symbol in symbols:
            self.assertEqual(len(symbol.notation.slurs), config.structured_slur_slots)

    def test_choices_are_attached_alongside_notation(self) -> None:
        symbols = _generate(_config(enable=True))

        self.assertTrue(all(symbol.structured_choices for symbol in symbols))

    def test_only_offered_heads_can_be_uncertain(self) -> None:
        # An untrained model is uncertain about everything, which makes this the
        # strongest available check that the policy filters by head rather than by
        # confidence alone: stems and ties must stay silent even here.
        symbols = _generate(_config(enable=True))

        surfaced = {
            choice.head
            for symbol in symbols
            for choice in symbol.structured_choices
            if choice.is_uncertain
        }

        self.assertTrue(surfaced, "an untrained model should be uncertain somewhere")
        for head in surfaced:
            self.assertTrue(
                head.startswith(("beam.level.", "slur.slot.")),
                f"{head} was surfaced to the user but is not an offered head",
            )

    def test_dynamics_are_never_written(self) -> None:
        symbols = _generate(_config(enable=True))

        for symbol in symbols:
            self.assertEqual(str(symbol.notation.dynamic), "none")

    def test_structured_choices_default_empty_without_heads(self) -> None:
        symbols = _generate(_config(enable=False))

        self.assertTrue(all(symbol.structured_choices == () for symbol in symbols))


if __name__ == "__main__":
    unittest.main()
