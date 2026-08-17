"""
The evaluation entry point, end to end on real objects.

Three defects in this work lived in entry-point bodies that no test exercised: a renderer
referencing unimported names, head weights that predated a head, and a statement ordered
before the value it depended on. Each was caught by running the thing for real, after the
unit tests had passed.

`test_structured_training_integration.py` does this for training, and that path has been
solid since. This is the same treatment for evaluation: a real TrOMR shrunk to one layer,
real images through the real loader, real weights saved and loaded back, and the manifest
deciding what may be missing.
"""

import json
import tempfile
import unittest
from argparse import Namespace
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
from training.transformer.evaluate_structured_heads import run_evaluation
from training.transformer.train_structured_heads import write_manifest
from training.transformer.training_vocabulary import token_lines_to_str


def _small_config() -> Config:
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


def _corpus(directory: Path) -> Path:
    notation = NoteNotation(
        beam_levels=(BeamLevelState.BEGIN, BeamLevelState.BEGIN) + empty_beam_levels()[2:],
        stem=StemDirection.UP,
        slurs=((SlurEvent.START, SlurSide.ABOVE),) + empty_slur_slots()[1:],
    )
    closing = NoteNotation(
        beam_levels=(BeamLevelState.END, BeamLevelState.END) + empty_beam_levels()[2:],
        stem=StemDirection.DOWN,
        slurs=((SlurEvent.STOP, SlurSide.BELOW),) + empty_slur_slots()[1:],
    )
    lines = []
    for number in range(2):
        image = directory / f"staff{number}.png"
        picture = np.full((64, 400, 3), 255, dtype=np.uint8)
        for line in range(5):
            picture[20 + line * 6, :, :] = 0
        cv2.imwrite(str(image), picture)

        symbols = [
            EncodedSymbol("clef_G2"),
            EncodedSymbol("note_16", "C5", notation=notation),
            EncodedSymbol("note_16", "D5", notation=closing),
            EncodedSymbol("barline"),
        ]
        tokens = directory / f"staff{number}.txt"
        tokens.write_text(token_lines_to_str(symbols), encoding="utf-8")
        write_sidecar(tokens, symbols)
        lines.append(f"{image},{tokens}")

    index = directory / "index.txt"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


class TestEvaluationEntryPoint(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.config = _small_config()

    def _args(self, directory: Path, **overrides: object) -> Namespace:
        model = TrOMR(self.config)
        checkpoint = directory / "core.pth"
        torch.save(model.state_dict(), checkpoint)
        weights = directory / "heads.pth"
        torch.save(model.decoder.structured_heads.state_dict(), weights)
        manifest = directory / "manifest.json"
        write_manifest(manifest, self.config, ("beam.level.1",), "a", "b", "run")

        defaults = {
            "index": _corpus(directory),
            "checkpoint": checkpoint,
            "weights": weights,
            "manifest": manifest,
            "out": None,
            "predictions": None,
            "batch_size": 2,
            "workers": 0,
            "device": "cpu",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_the_whole_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            evaluation = run_evaluation(self._args(directory), self.config)

            self.assertEqual(evaluation.sequences, 2)

    def test_only_the_manifest_declared_head_is_scored(self) -> None:
        # The manifest names one beam level, so nothing else may contribute a figure.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)

            evaluation = run_evaluation(self._args(directory), self.config)

            self.assertTrue(evaluation.per_level[1].classes)
            self.assertFalse(any(m.support for m in evaluation.stems.classes.values()))

    def test_the_report_and_predictions_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = directory / "report.json"
            predictions = directory / "predictions.jsonl"

            run_evaluation(
                self._args(directory, out=report, predictions=predictions), self.config
            )

            self.assertIn("exact_beam_vector", json.loads(report.read_text(encoding="utf-8")))
            self.assertEqual(len(predictions.read_text(encoding="utf-8").splitlines()), 2)

    def test_weights_missing_a_scored_head_are_refused(self) -> None:
        # The version-skew guard, reached through the entry point rather than directly.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            args = self._args(directory)
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            torch.save({k: v for k, v in state.items() if not k.startswith("beam.")}, args.weights)

            with self.assertRaises(ValueError):
                run_evaluation(args, self.config)


if __name__ == "__main__":
    unittest.main()
