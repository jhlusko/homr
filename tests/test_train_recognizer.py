import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

from training.architecture.ocr.crnn import Alphabet, CRNN
from training.ocr.recognizer_data import SyllableCrops, collate, read_manifest
from training.ocr.train_recognizer import Accuracy, edit_distance, evaluate, train
from torch.utils.data import DataLoader


def _corpus(directory: Path, name: str, rows: list[str]) -> Path:
    path = directory / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(rows):
            image = directory / f"{name}{index}.png"
            picture = np.full((48, 16 * max(2, len(text))), 255, dtype=np.uint8)
            picture[20:30, 4:12] = 0
            cv2.imwrite(str(image), picture)
            handle.write(json.dumps({"image": str(image), "text": text, "score": name}) + "\n")
    return path


class TestEditDistance(unittest.TestCase):
    def test_identical_strings_are_zero_apart(self) -> None:
        self.assertEqual(edit_distance("gues", "gues"), 0)

    def test_one_substitution_costs_one(self) -> None:
        self.assertEqual(edit_distance("gues", "gnes"), 1)

    def test_an_empty_prediction_costs_the_whole_word(self) -> None:
        self.assertEqual(edit_distance("gues", ""), 4)


class TestAccuracy(unittest.TestCase):
    def test_exact_and_cer_are_both_tracked(self) -> None:
        accuracy = Accuracy()
        accuracy.observe("va", "va")
        accuracy.observe("gues", "gnes")

        self.assertEqual(accuracy.exact, 1)
        self.assertAlmostEqual(accuracy.distance / accuracy.characters, 1 / 6)

    def test_misreads_are_kept_for_inspection(self) -> None:
        # A rate says how often; the examples say what kind, which is what suggests a fix.
        accuracy = Accuracy()
        accuracy.observe("gues", "gnes")

        self.assertEqual(accuracy.examples, [("gues", "gnes")])

    def test_nothing_scored_is_reported_rather_than_divided_by_zero(self) -> None:
        self.assertIn("nothing", Accuracy().describe("seen"))


class TestSeenUnseenSplit(unittest.TestCase):
    """17.0% of validation syllables never appear in training, so a model that memorised
    the training vocabulary and read nothing would still score 83%."""

    def test_each_syllable_lands_on_the_right_side(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = read_manifest(_corpus(directory, "v", ["va", "novel"]))
            alphabet = Alphabet("vanoel")
            model = CRNN(len(alphabet))
            dataset = SyllableCrops(samples, alphabet, model.frame_count)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate)

            known, novel = evaluate(model, loader, alphabet, {"va"}, "cpu")

        self.assertEqual(known.total, 1)
        self.assertEqual(novel.total, 1)

    def test_an_empty_side_does_not_crash_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = read_manifest(_corpus(directory, "v", ["va"]))
            alphabet = Alphabet("va")
            model = CRNN(len(alphabet))
            loader = DataLoader(
                SyllableCrops(samples, alphabet, model.frame_count),
                batch_size=2, collate_fn=collate,
            )

            known, novel = evaluate(model, loader, alphabet, {"va"}, "cpu")

        self.assertEqual(novel.total, 0)
        self.assertIn("nothing", novel.describe("unseen"))


class TestCropHeight(unittest.TestCase):
    """32 was taken from scene-text convention without measuring, and downscales 93% of
    crops - discarding the detail 27.47's sampled render resolution exists to provide."""

    def test_the_recurrent_layer_follows_the_height(self) -> None:
        import torch

        for height in (32, 48, 64):
            with self.subTest(height=height):
                model = CRNN(8, image_height=height)
                output = model(torch.rand(1, 1, height, 64))
                self.assertEqual(output.shape[2], 8)

    def test_a_height_the_stack_cannot_halve_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            CRNN(8, image_height=40)


class TestTrainEntryPoint(unittest.TestCase):
    """The three defects of this project that reached production all lived in entry points
    no test executed."""

    def _args(self, directory: Path, **overrides) -> Namespace:
        defaults = {
            "train": _corpus(directory, "train", ["va", "gues", "des", "va"]),
            "valid": _corpus(directory, "valid", ["va", "novel"]),
            "weights": directory / "w.pth",
            "out": directory / "history.json",
            "height": 32,
            "epochs": 1,
            "lr": 3e-4,
            "batch_size": 2,
            "workers": 0,
            "device": "cpu",
        }
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_the_whole_path_runs_and_reports_both_sides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = train(self._args(Path(tmp)))

        self.assertEqual(len(report["history"]), 1)
        self.assertIn("unseen_exact", report["history"][0])
        self.assertIn("seen_exact", report["history"][0])

    def test_the_alphabet_comes_from_training_only(self) -> None:
        # A character seen only in validation is one the model could never emit; counting
        # it would inflate the output layer while pretending it is learnable.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            report = train(
                self._args(directory, valid=_corpus(directory, "valid", ["va", "Zzz"]))
            )

        self.assertNotIn("Z", report["alphabet"])

    def test_weights_are_written_with_their_alphabet(self) -> None:
        # Weights without the alphabet that indexed them decode to nonsense.
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            train(self._args(directory))
            saved = torch.load(directory / "w.pth", map_location="cpu", weights_only=False)

        self.assertIn("alphabet", saved)
        self.assertIn("model", saved)


if __name__ == "__main__":
    unittest.main()
