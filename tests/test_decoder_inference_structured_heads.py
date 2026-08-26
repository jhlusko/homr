"""Structured-head wiring in the production ONNX inference path.

`homr/transformer/staff2score.Staff2Score` - what `homr/main.py` actually runs - reaches
`ScoreDecoder.generate()` in `decoder_inference.py`, not the training-side torch decoder.
That loop already computed `hidden` (index 7 of `output_names`) but never read it, so the
structured heads were wired into the training-side `generate()` and still reached no
production output at all. These tests pin the fix.

A fake session stands in for the real ONNX head graph: the point here is the plumbing
(hidden reaches the heads, in the right shape, with the right cast, only for a step the
model actually decided), not re-deriving `decode_note`'s own decisions, which
`test_structured_decode.py` already covers, or the real head weights' numeric behaviour,
which `test_convert_structured_heads.py` covers against the exported graph.
"""

import math
import unittest

import numpy as np
import onnxruntime as ort

from homr.transformer.configs import Config
from homr.transformer.decoder_inference import ScoreDecoder
from homr.transformer.structured_notation import BeamLevelState


class FakeHeadsSession:
    """Stands in for an onnxruntime session over the exported structured-heads graph.

    Returns a fixed, strongly-peaked logit vector for `beam.level.1` so the resulting
    decode is deterministic and checkable, and empty/uniform vectors for the rest -
    enough to exercise the plumbing without needing the real trained weights.
    """

    NAMES = ["beam.level.1", "stem.direction"]

    def get_outputs(self):
        return [type("Output", (), {"name": name})() for name in self.NAMES]

    def run(self, output_names, feeds):
        hidden = feeds["hidden"]
        self.last_feed_dtype = hidden.dtype
        self.last_feed_shape = hidden.shape
        seq_len = hidden.shape[1]
        beam_classes = list(BeamLevelState)
        begin_index = beam_classes.index(BeamLevelState.BEGIN)
        beam_logits = np.full((1, seq_len, len(beam_classes)), -10.0, dtype=np.float32)
        beam_logits[:, :, begin_index] = 10.0
        stem_logits = np.zeros((1, seq_len, 5), dtype=np.float32)
        by_name = {"beam.level.1": beam_logits, "stem.direction": stem_logits}
        return [by_name[name] for name in output_names]


def _bare_decoder() -> ScoreDecoder:
    """A ScoreDecoder with no real ONNX transformer - only what _attach_structured_notation
    and __init__ touch are exercised, so a real decoder graph is not needed."""
    config = Config()
    return ScoreDecoder.__new__(ScoreDecoder)


class TestAttachStructuredNotation(unittest.TestCase):
    def decoder(self, heads=None) -> ScoreDecoder:
        instance = _bare_decoder()
        instance.structured_heads = heads
        instance.structured_heads_names = (
            [o.name for o in heads.get_outputs()] if heads is not None else []
        )
        return instance

    def test_no_session_leaves_the_symbol_untouched(self) -> None:
        from homr.transformer.vocabulary import EncodedSymbol

        instance = self.decoder(heads=None)
        symbol = EncodedSymbol("note_8")
        hidden = np.zeros((1, 1, 512), dtype=np.float32)

        instance._attach_structured_notation(symbol, hidden)

        self.assertIsNone(symbol.notation)
        self.assertEqual(symbol.structured_choices, ())

    def test_a_session_populates_notation_from_the_last_token(self) -> None:
        from homr.transformer.vocabulary import EncodedSymbol

        fake = FakeHeadsSession()
        instance = self.decoder(heads=fake)
        symbol = EncodedSymbol("note_8")
        hidden = np.zeros((1, 3, 512), dtype=np.float32)  # 3 tokens; only the last matters

        instance._attach_structured_notation(symbol, hidden)

        self.assertEqual(symbol.notation.beam_levels[0], BeamLevelState.BEGIN)
        self.assertEqual(fake.last_feed_shape[1], 1, "must feed only the last token, not all 3")

    def test_hidden_is_cast_to_float32_before_reaching_the_session(self) -> None:
        # The decoder can run fp16 on the GPU EP; the heads graph was only ever
        # exported fp32. Feeding fp16 straight in would silently mismatch the graph's
        # declared input dtype.
        fake = FakeHeadsSession()
        instance = self.decoder(heads=fake)
        from homr.transformer.vocabulary import EncodedSymbol

        symbol = EncodedSymbol("note_8")
        hidden = np.zeros((1, 1, 512), dtype=np.float16)

        instance._attach_structured_notation(symbol, hidden)

        self.assertEqual(fake.last_feed_dtype, np.float32)

    def test_uncertain_beam_choices_are_attached_when_present(self) -> None:
        from homr.transformer.vocabulary import EncodedSymbol

        class UncertainHeads(FakeHeadsSession):
            def run(self, output_names, feeds):
                hidden = feeds["hidden"]
                self.last_feed_dtype = hidden.dtype
                self.last_feed_shape = hidden.shape
                beam_classes = list(BeamLevelState)
                end = beam_classes.index(BeamLevelState.END)
                cont = beam_classes.index(BeamLevelState.CONTINUE)
                logits = np.full((1, 1, len(beam_classes)), -10.0, dtype=np.float32)
                logits[:, :, end] = math.log(0.52)
                logits[:, :, cont] = math.log(0.48)
                stem_logits = np.zeros((1, 1, 5), dtype=np.float32)
                by_name = {"beam.level.1": logits, "stem.direction": stem_logits}
                return [by_name[name] for name in output_names]

        instance = self.decoder(heads=UncertainHeads())
        symbol = EncodedSymbol("note_8")

        instance._attach_structured_notation(symbol, np.zeros((1, 1, 512), dtype=np.float32))

        self.assertTrue(any(c.is_uncertain for c in symbol.structured_choices))


class TestGetDecoderLoadsHeadsOptionally(unittest.TestCase):
    def test_no_heads_file_means_no_session(self) -> None:
        from unittest import mock

        from homr.transformer import decoder_inference

        config = Config()
        config.use_gpu_inference = False
        config.filepaths.decoder_path = "/dev/null"
        config.filepaths.structured_heads_path = "/definitely/does/not/exist.onnx"

        with mock.patch.object(ort, "InferenceSession") as fake_session:
            fake_session.return_value = mock.Mock()
            result = decoder_inference.get_decoder(config)

        self.assertIsNone(result.structured_heads)

    def test_a_present_heads_file_is_loaded(self) -> None:
        from unittest import mock

        from homr.transformer import decoder_inference

        config = Config()
        config.use_gpu_inference = False
        config.filepaths.decoder_path = "/dev/null"
        config.filepaths.structured_heads_path = __file__  # any real, existing path

        with mock.patch.object(ort, "InferenceSession") as fake_session:
            fake_session.return_value.get_outputs.return_value = [
                type("Output", (), {"name": "beam.level.1"})()
            ]
            result = decoder_inference.get_decoder(config)

        # Called once for the decoder, once for the heads.
        self.assertEqual(fake_session.call_count, 2)
        self.assertIsNotNone(result.structured_heads)


if __name__ == "__main__":
    unittest.main()
