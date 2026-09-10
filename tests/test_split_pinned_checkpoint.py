import tempfile
import unittest
from pathlib import Path

import torch

from training.onnx.split_pinned_checkpoint import split_checkpoint


class TestSplitCheckpoint(unittest.TestCase):
    def state(self) -> dict:
        return {
            "encoder.model.head.norm.bias": torch.zeros(2),
            "encoder.model.stem.weight": torch.ones(3),
            "decoder.net.articulation_emb.emb.weight": torch.full((2,), 5.0),
            "decoder.note_mask": torch.tensor([1.0, 0.0]),
        }

    def test_encoder_keys_lose_their_prefix(self) -> None:
        encoder, _ = split_checkpoint(self.state())

        self.assertEqual(
            set(encoder), {"model.head.norm.bias", "model.stem.weight"}
        )

    def test_decoder_net_keys_lose_the_full_net_prefix(self) -> None:
        _, decoder = split_checkpoint(self.state())

        self.assertEqual(set(decoder), {"articulation_emb.emb.weight"})

    def test_decoder_attributes_outside_net_are_dropped(self) -> None:
        # decoder.note_mask is ScoreDecoder's own buffer, not something
        # ScoreTransformerWrapper (what convert_decoder actually loads) can accept.
        _, decoder = split_checkpoint(self.state())

        self.assertNotIn("note_mask", decoder)

    def test_values_are_preserved_not_just_keys(self) -> None:
        encoder, decoder = split_checkpoint(self.state())

        self.assertTrue(torch.equal(encoder["model.stem.weight"], torch.ones(3)))
        self.assertTrue(
            torch.equal(decoder["articulation_emb.emb.weight"], torch.full((2,), 5.0))
        )

    def test_every_input_tensor_lands_in_exactly_one_place_or_neither(self) -> None:
        state = self.state()
        encoder, decoder = split_checkpoint(state)

        self.assertEqual(len(encoder) + len(decoder) + 1, len(state))  # +1 = note_mask

    def test_an_empty_checkpoint_splits_into_two_empty_dicts(self) -> None:
        encoder, decoder = split_checkpoint({})

        self.assertEqual((encoder, decoder), ({}, {}))

    def test_main_writes_both_files(self) -> None:
        from training.onnx.split_pinned_checkpoint import main

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(self.state(), checkpoint)
            out = Path(directory) / "out"

            import sys

            argv = sys.argv
            sys.argv = [
                "split_pinned_checkpoint.py",
                "--checkpoint", str(checkpoint),
                "--out", str(out),
            ]
            try:
                main()
            finally:
                sys.argv = argv

            self.assertTrue((out / "encoder_weights.pt").exists())
            self.assertTrue((out / "decoder_weights.pt").exists())
            loaded = torch.load(out / "decoder_weights.pt", weights_only=True)
            self.assertEqual(set(loaded), {"articulation_emb.emb.weight"})


if __name__ == "__main__":
    unittest.main()
