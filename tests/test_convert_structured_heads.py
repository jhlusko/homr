"""The structured heads must survive ONNX export with their decisions intact.

Production inference runs ONNX, not PyTorch, so heads that train well and export badly
would be heads that never reach a user. Export succeeding proves nothing on its own -
what matters is that the exported graph makes the *same notation decisions* as the torch
module, which is what these tests check.

The heads export as their own graph rather than as extra decoder outputs: they are a
non-autoregressive projection of the hidden state, and the decoder graph already emits
`hidden`. Nothing about the decoder export changes to carry them.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from homr.transformer.configs import Config
from homr.transformer.structured_decode import decode_note
from training.architecture.transformer.structured_heads import StructuredNotationHeads

try:
    import onnxruntime as ort

    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

from training.onnx.convert import StructuredHeadsWrapper


def _heads(config: Config) -> StructuredNotationHeads:
    heads = StructuredNotationHeads(
        dim=config.decoder_dim,
        beam_levels=config.structured_beam_levels,
        slur_slots=config.structured_slur_slots,
    )
    heads.eval()
    return heads


def _export(heads: StructuredNotationHeads, config: Config, path: Path) -> list[str]:
    names = sorted(heads.head_names())
    torch.onnx.export(
        StructuredHeadsWrapper(heads, names),
        (torch.randn((1, 1, config.decoder_dim)).float(),),
        str(path),
        input_names=["hidden"],
        output_names=names,
        dynamic_axes={"hidden": {1: "seq_len"}, **{n: {1: "seq_len"} for n in names}},
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        dynamo=False,
    )
    return names


class TestWrapperOrdering(unittest.TestCase):
    def test_outputs_follow_the_declared_name_order(self) -> None:
        # ONNX outputs are positional, so head order is part of the file's contract.
        # Dict order would make it depend on construction details.
        config = Config()
        heads = _heads(config)
        names = sorted(heads.head_names())
        hidden = torch.randn(1, 1, config.decoder_dim)

        with torch.no_grad():
            wrapped = StructuredHeadsWrapper(heads, names)(hidden)
            direct = heads(hidden)

        for index, name in enumerate(names):
            self.assertTrue(torch.equal(wrapped[index], direct[name]))

    def test_the_order_is_stable_across_constructions(self) -> None:
        config = Config()

        self.assertEqual(sorted(_heads(config).head_names()), sorted(_heads(config).head_names()))


@unittest.skipUnless(_HAS_ORT, "onnxruntime not installed")
class TestExportedGraphAgrees(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config()
        cls.heads = _heads(cls.config)
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "heads.onnx"
        cls.names = _export(cls.heads, cls.config, cls.path)
        cls.session = ort.InferenceSession(
            str(cls.path), providers=["CPUExecutionProvider"]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_it_exports_every_head(self) -> None:
        self.assertEqual(
            {output.name for output in self.session.get_outputs()}, set(self.names)
        )

    def test_logits_match_torch(self) -> None:
        torch.manual_seed(0)
        worst = 0.0
        for _ in range(10):
            hidden = torch.randn(1, 1, self.config.decoder_dim)
            with torch.no_grad():
                reference = self.heads(hidden)
            produced = self.session.run(self.names, {"hidden": hidden.numpy()})
            for name, array in zip(self.names, produced, strict=True):
                worst = max(
                    worst, (reference[name] - torch.from_numpy(array)).abs().max().item()
                )

        self.assertLess(worst, 1e-4, f"ONNX diverged from torch by {worst}")

    def test_the_decoded_notation_is_identical(self) -> None:
        # The number that actually matters. Small logit drift is harmless; a different
        # decoded beam state is a different score.
        torch.manual_seed(1)
        for _ in range(25):
            hidden = torch.randn(1, 1, self.config.decoder_dim)
            with torch.no_grad():
                reference = self.heads(hidden)
            produced = self.session.run(self.names, {"hidden": hidden.numpy()})

            from_torch = decode_note(
                {name: reference[name][0, -1, :].tolist() for name in self.names}
            )
            from_onnx = decode_note(
                {
                    name: array[0, -1, :].tolist()
                    for name, array in zip(self.names, produced, strict=True)
                }
            )

            self.assertEqual(from_torch.notation, from_onnx.notation)

    def test_a_multi_token_hidden_state_is_accepted(self) -> None:
        # The decoder feeds one token per step, but the sequence axis is dynamic so the
        # same file can score a whole staff at once.
        hidden = torch.randn(1, 7, self.config.decoder_dim)

        produced = self.session.run(self.names, {"hidden": hidden.numpy()})

        self.assertEqual(produced[0].shape[1], 7)


if __name__ == "__main__":
    unittest.main()
