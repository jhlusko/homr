"""
The whole frozen-core chain on real objects, once.

Every other test for this work uses a stand-in somewhere: a fake dataset, a fake model, a
hand-built batch. That is right for testing one decision at a time, and it means nothing
has ever run an index file through the real loader, into the real model, out through the
real heads and into the real loss. The failures that live in those seams - a key the
collate does not expect, a hidden state one position out, a head whose logits do not
reach the optimiser - are exactly the ones a stand-in cannot show.

The model here is a real TrOMR, shrunk to one layer so it fits in a test. The images are
real PNGs read by the real image pipeline, and the token files carry real sidecars.
"""

import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from homr.transformer.configs import Config
from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.tromr_arch import TrOMR
from training.omr_datasets.notation_sidecar import write_sidecar
from training.transformer.train_structured_heads import (
    _target_names,
    build_batches,
    structured_parameters,
    train_epoch,
)
from training.transformer.training_vocabulary import token_lines_to_str


def _small_config() -> Config:
    """A real Config, shrunk everywhere that does not change the interfaces.

    The canvas size is left alone: the loader centres every staff on a fixed 256x1280
    canvas regardless of config, so shrinking it here would only make the encoder reject
    its own input.
    """
    config = Config()
    config.enable_structured_heads = True
    config.encoder_depth = 1
    config.decoder_depth = 1
    config.encoder_dim = 96
    config.encoder_h_dim = config.encoder_dim // 3
    config.encoder_heads = 4
    config.decoder_dim = config.encoder_dim
    config.decoder_heads = 4
    config.backbone_layers = [1, 1, 1, 1]
    return config


def _staff_image(path: Path) -> None:
    """Something with staff-like structure rather than noise, so the encoder sees lines."""
    image = np.full((64, 400, 3), 255, dtype=np.uint8)
    for line in range(5):
        image[20 + line * 6, :, :] = 0
    cv2.imwrite(str(path), image)


def _tokens(path: Path) -> None:
    notation = NoteNotation(
        beam_levels=(BeamLevelState.BEGIN, BeamLevelState.BEGIN) + empty_beam_levels()[2:],
        stem=StemDirection.UP,
        slurs=((SlurEvent.START, SlurSide.ABOVE),) + empty_slur_slots()[1:],
    )
    closing = NoteNotation(
        beam_levels=(BeamLevelState.END, BeamLevelState.END) + empty_beam_levels()[2:],
        stem=StemDirection.DOWN,
        slurs=((SlurEvent.STOP, SlurSide.ABOVE),) + empty_slur_slots()[1:],
    )
    symbols = [
        EncodedSymbol("clef_G2"),
        EncodedSymbol("note_16", "C5", notation=notation),
        EncodedSymbol("note_16", "D5", notation=closing),
        EncodedSymbol("barline"),
    ]
    path.write_text(token_lines_to_str(symbols), encoding="utf-8")
    write_sidecar(path, symbols)


def _corpus(directory: Path, examples: int = 2) -> Path:
    index = directory / "index.txt"
    lines = []
    for number in range(examples):
        image = directory / f"staff{number}.png"
        tokens = directory / f"staff{number}.txt"
        _staff_image(image)
        _tokens(tokens)
        lines.append(f"{image},{tokens}")
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


class TestFrozenCoreEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = _small_config()
        self.model = TrOMR(self.config)
        self.trainable = self.model.freeze_core_for_structured_heads()
        self.names = _target_names(self.config)

    def _run(self, directory: str) -> tuple:
        index = _corpus(Path(directory))
        batches, count = build_batches(index, self.config, batch_size=2, workers=0)
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.05)
        report = train_epoch(self.model, batches, optimizer, self.names, epoch=1)
        return report, count

    def test_the_chain_runs_and_the_heads_are_supervised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report, count = self._run(tmp)

        self.assertEqual(count, 2)
        self.assertGreater(report.batches, 0)
        # Both notes are 16ths, so levels 1 and 2 apply and 3 and 4 do not. A run where
        # everything reported support would mean the masking never reached the loss.
        self.assertGreater(report.support["beam.level.1"], 0)
        self.assertGreater(report.support["stem.direction"], 0)
        self.assertEqual(report.support["beam.level.4"], 0)

    def test_the_frozen_core_does_not_move(self) -> None:
        # The claim the whole experiment rests on, checked against the real model rather
        # than a stand-in with one linear layer.
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if not name.startswith("decoder.structured_heads.")
        }

        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)

        for name, parameter in self.model.named_parameters():
            if name in before:
                self.assertTrue(torch.equal(before[name], parameter), f"{name} moved")

    def test_the_heads_receive_gradient(self) -> None:
        # A head can be listed as trainable and still never be reached - by a detached
        # hidden state, or by a loss that never includes it.
        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp)

        heads = self.model.decoder.structured_heads
        touched = [name for name, p in heads.named_parameters() if p.grad is not None]
        self.assertTrue(touched)

    def test_the_existing_objective_is_untouched(self) -> None:
        # The structured loss must never join `loss`: B0 has to stay comparable.
        with tempfile.TemporaryDirectory() as tmp:
            index = _corpus(Path(tmp))
            batches, _ = build_batches(index, self.config, batch_size=2, workers=0)
            batch = next(iter(batches))
            inputs = {k: v for k, v in batch.items() if k not in self.names}
            outputs = self.model(**inputs)

        self.assertIn("loss", outputs)
        self.assertIn("structured_logits", outputs)
        self.assertTrue(torch.isfinite(outputs["loss"]))


if __name__ == "__main__":
    unittest.main()
