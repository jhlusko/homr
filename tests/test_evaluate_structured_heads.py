import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    STEM_CLASSES,
    BeamLevelState,
    StemDirection,
)
from training.architecture.transformer.structured_heads import StructuredNotationHeads
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.transformer.evaluate_structured_heads import evaluate, trained_heads
from training.transformer.train_structured_heads import write_manifest

HEADS = ["beam.level.1", "beam.level.2", "stem.direction", "slur.slot.1.event", "slur.slot.1.side"]


class _Config:
    max_height = 256
    max_width = 1280
    max_seq_len = 608
    structured_beam_levels = 2
    structured_slur_slots = 1


class _Model(nn.Module):
    """Emits a fixed class for every head at every position.

    A constant answer is exactly the failure the metrics exist to catch, so it is the
    right stand-in: if the pipeline scores it well, the pipeline is broken.
    """

    def __init__(self, choices: dict[str, int]) -> None:
        super().__init__()
        self.choices = choices
        self.decoder = nn.Module()
        self.decoder.structured_heads = StructuredNotationHeads(dim=8, beam_levels=2, slur_slots=1)

    def forward(self, **batch: torch.Tensor) -> dict:
        # The real decoder refuses unexpected keyword arguments, which is how a target key
        # leaking into the model call surfaced. Reproduced here so the test can catch it.
        unexpected = [key for key in batch if key not in ("inputs",)]
        if unexpected:
            raise TypeError(f"unexpected keyword argument {unexpected[0]!r}")
        hidden = batch["inputs"][:, :-1]
        logits = self.decoder.structured_heads(hidden)
        fixed = {}
        for name, tensor in logits.items():
            picked = torch.full_like(tensor, -10.0)
            picked[..., self.choices.get(name, 0)] = 10.0
            fixed[name] = picked
        return {"structured_logits": fixed}


def _batch(stem: StemDirection, beam: BeamLevelState) -> dict:
    """One sequence of six positions, supervised at position 3 only."""
    blank = torch.full((1, 6), IGNORE_INDEX, dtype=torch.long)
    batch = {"inputs": torch.randn(1, 6, 8)}
    for name in HEADS:
        batch[name] = blank.clone()
    batch["stem.direction"][:, 3] = STEM_CLASSES.index(stem)
    batch["beam.level.1"][:, 3] = BEAM_LEVEL_CLASSES.index(beam)
    return batch


class TestEvaluate(unittest.TestCase):
    def test_a_correct_head_scores_perfectly(self) -> None:
        model = _Model(
            {
                "stem.direction": STEM_CLASSES.index(StemDirection.UP),
                "beam.level.1": BEAM_LEVEL_CLASSES.index(BeamLevelState.BEGIN),
            }
        )
        batches = [_batch(StemDirection.UP, BeamLevelState.BEGIN)]

        result = evaluate(model, batches, HEADS, beam_levels=2, slur_slots=1)

        self.assertEqual(result.stems.macro_f1, 1.0)
        self.assertEqual(result.exact_vector_rate, 1.0)

    def test_a_head_answering_one_class_everywhere_is_caught(self) -> None:
        model = _Model({"stem.direction": STEM_CLASSES.index(StemDirection.UP)})
        batches = [
            _batch(StemDirection.UP, BeamLevelState.BEGIN),
            _batch(StemDirection.DOWN, BeamLevelState.BEGIN),
        ]

        result = evaluate(model, batches, HEADS, beam_levels=2, slur_slots=1)

        # Right half the time by luck, and the macro average must not read as 0.5.
        self.assertLess(result.stems.macro_f1, 0.5)

    def test_masked_positions_are_not_scored(self) -> None:
        # Five of six positions carry no target. If they were scored the constant model
        # would look far better than it is.
        model = _Model({"stem.direction": STEM_CLASSES.index(StemDirection.UP)})

        result = evaluate(model, [_batch(StemDirection.UP, BeamLevelState.BEGIN)], HEADS, 2, 1)

        support = sum(m.support for m in result.stems.classes.values())
        self.assertEqual(support, 1)

    def test_a_batch_with_no_targets_is_skipped(self) -> None:
        model = _Model({})

        result = evaluate(model, [{"inputs": torch.randn(1, 6, 8)}], HEADS, 2, 1)

        self.assertEqual(result.sequences, 0)


class TestTrainedHeads(unittest.TestCase):
    def _manifest(self, tmp: str, supported: list[str]) -> Path:
        # Written through the real builder rather than hand-rolled, so the fixture cannot
        # drift from the schema the reader enforces.
        path = Path(tmp) / "manifest.json"
        write_manifest(path, _Config(), tuple(supported), "a", "b", "r")
        return path

    def test_only_declared_heads_are_scored(self) -> None:
        # An untrained projection still emits an argmax; reporting a number for it would
        # claim a capability that does not exist.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest(tmp, ["beam.level.1"])

            self.assertEqual(trained_heads(path, HEADS), ["beam.level.1"])

    def test_without_a_manifest_everything_is_scored(self) -> None:
        self.assertEqual(trained_heads(None, HEADS), HEADS)


class TestUnscoredHeadsAreStillStrippedFromTheInput(unittest.TestCase):
    """A head the manifest does not declare still has target tensors in the batch.

    Every key that is not a target gets forwarded to the model as a keyword argument, so
    scoring seven heads while the batch carries nine hands the decoder `slur.slot.1.side`
    and it refuses it. This is what killed the first evaluation run after a successful
    training run.
    """

    def test_scoring_a_subset_does_not_leak_the_rest_into_the_model(self) -> None:
        model = _Model({})
        declared = ["beam.level.1", "stem.direction"]

        result = evaluate(
            model,
            [_batch(StemDirection.UP, BeamLevelState.BEGIN)],
            declared,
            beam_levels=2,
            slur_slots=1,
            all_targets=HEADS,
        )

        self.assertEqual(result.sequences, 1)

    def test_without_all_targets_it_falls_back_to_what_is_scored(self) -> None:
        # The training path passes one list because it scores everything it labels.
        model = _Model({})

        result = evaluate(
            model, [_batch(StemDirection.UP, BeamLevelState.BEGIN)], HEADS, 2, 1
        )

        self.assertEqual(result.sequences, 1)


if __name__ == "__main__":
    unittest.main()
