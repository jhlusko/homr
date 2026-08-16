import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from homr.transformer.capability_manifest import CapabilityManifest
from training.architecture.transformer.structured_heads import StructuredNotationHeads
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.transformer.train_structured_heads import (
    EpochReport,
    heads_with_support,
    structured_parameters,
    train_epoch,
    write_manifest,
)


class _Config:
    max_height = 256
    max_width = 1280
    max_seq_len = 608
    structured_beam_levels = 2
    structured_slur_slots = 1


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.structured_heads = StructuredNotationHeads(dim=8, beam_levels=2, slur_slots=1)


class _Model(nn.Module):
    """A stand-in with the shape train_epoch depends on: a frozen body, the heads, and a
    forward returning structured_logits."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(8, 8)
        self.decoder = _Decoder()

    def forward(self, **batch: torch.Tensor) -> dict:
        hidden = self.body(batch["inputs"])
        return {"structured_logits": self.decoder.structured_heads(hidden)}


def _batch(supervised: bool = True) -> dict:
    targets = torch.full((2, 5), IGNORE_INDEX, dtype=torch.long)
    if supervised:
        targets[:, 2] = 1
    return {
        "inputs": torch.randn(2, 5, 8),
        "beam.level.1": targets.clone(),
        "beam.level.2": torch.full((2, 5), IGNORE_INDEX, dtype=torch.long),
        "stem.direction": targets.clone(),
        "slur.slot.1.event": targets.clone(),
        "slur.slot.1.side": torch.full((2, 5), IGNORE_INDEX, dtype=torch.long),
    }


HEADS = ["beam.level.1", "beam.level.2", "stem.direction", "slur.slot.1.event", "slur.slot.1.side"]


class TestFrozenCoreStep(unittest.TestCase):
    def setUp(self) -> None:
        self.model = _Model()
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith("decoder.structured_heads.")

    def test_only_the_heads_are_optimisable(self) -> None:
        params = structured_parameters(self.model)

        self.assertTrue(params)
        self.assertEqual(len(params), len(list(self.model.decoder.structured_heads.parameters())))

    def test_the_frozen_body_does_not_move(self) -> None:
        # The whole experiment rests on this: a gain must be the heads, not the core
        # drifting to suit them.
        before = self.model.body.weight.detach().clone()
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.1)

        train_epoch(self.model, [_batch() for _ in range(3)], optimizer, HEADS, epoch=1)

        self.assertTrue(torch.equal(before, self.model.body.weight))

    def test_the_heads_do_move(self) -> None:
        before = self.model.decoder.structured_heads.stem.weight.detach().clone()
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.1)

        train_epoch(self.model, [_batch() for _ in range(3)], optimizer, HEADS, epoch=1)

        self.assertFalse(torch.equal(before, self.model.decoder.structured_heads.stem.weight))

    def test_support_is_reported_per_head(self) -> None:
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.01)

        report = train_epoch(self.model, [_batch()], optimizer, HEADS, epoch=1)

        self.assertEqual(report.support["beam.level.1"], 2)
        # A head with no targets at all must read zero rather than being absent.
        self.assertEqual(report.support["beam.level.2"], 0)

    def test_a_batch_with_no_targets_is_skipped_not_averaged_in(self) -> None:
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.01)

        report = train_epoch(self.model, [{"inputs": torch.randn(2, 5, 8)}], optimizer, HEADS, 1)

        self.assertEqual(report.batches, 0)

    def test_a_head_that_never_saw_a_target_is_named(self) -> None:
        optimizer = torch.optim.Adam(structured_parameters(self.model), lr=0.01)

        report = train_epoch(self.model, [_batch()], optimizer, HEADS, epoch=1)

        self.assertIn("no targets all epoch", report.describe())
        self.assertIn("beam.level.2", report.describe())


class TestDeclaringWhatWasTrained(unittest.TestCase):
    def test_only_heads_with_support_are_declared(self) -> None:
        # A projection that never saw a gradient still emits logits; declaring it would
        # advertise a head holding its initialisation.
        reports = [
            EpochReport(1, 0.0, 1, {}, {"beam.level.1": 12, "beam.level.2": 0}),
            EpochReport(2, 0.0, 1, {}, {"beam.level.1": 9, "beam.level.2": 0}),
        ]

        self.assertEqual(heads_with_support(reports), ("beam.level.1",))

    def test_the_manifest_records_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_manifest(path, _Config(), ("beam.level.1",), "abc", "def", "run-1")
            manifest = CapabilityManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

        self.assertEqual(manifest.supported_heads, ("beam.level.1",))
        self.assertFalse(manifest.supports("stem.direction"))
        self.assertEqual(manifest.run_id, "run-1")


if __name__ == "__main__":
    unittest.main()
