import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from training.architecture.ocr.crnn import IMAGE_HEIGHT, Alphabet
from training.ocr.recognizer_data import (
    PAD_VALUE,
    Sample,
    SyllableCrops,
    alphabet_of,
    collate,
    read_manifest,
    scaled_width,
)


def _crop(directory: Path, name: str, width: int, height: int = 48) -> Path:
    path = directory / f"{name}.png"
    cv2.imwrite(str(path), np.full((height, width), 200, dtype=np.uint8))
    return path


def _manifest(directory: Path, rows: list[tuple[str, int]]) -> Path:
    path = directory / "m.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for text, width in rows:
            image = _crop(directory, text.replace("!", "x"), width)
            handle.write(json.dumps({"image": str(image), "text": text, "score": "s1"}) + "\n")
    return path


class TestManifest(unittest.TestCase):
    def test_rows_become_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            samples = read_manifest(_manifest(Path(tmp), [("va", 40), ("gues", 70)]))

        self.assertEqual([s.text for s in samples], ["va", "gues"])

    def test_blank_lines_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _manifest(Path(tmp), [("va", 40)])
            path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

            self.assertEqual(len(read_manifest(path)), 1)


class TestAlphabetOf(unittest.TestCase):
    def test_it_covers_every_character_in_the_corpus(self) -> None:
        samples = [Sample(Path("a"), "säu", ""), Sample(Path("b"), "Ê!", "")]

        alphabet = alphabet_of(samples)

        self.assertEqual(set(alphabet.characters), set("säuÊ!"))


class TestScaledWidth(unittest.TestCase):
    def test_the_aspect_ratio_is_kept(self) -> None:
        self.assertEqual(scaled_width(96, IMAGE_HEIGHT * 2), 48)

    def test_a_tiny_crop_still_has_some_width(self) -> None:
        self.assertGreaterEqual(scaled_width(1, 400), 8)


class TestSyllableCrops(unittest.TestCase):
    def test_a_label_too_long_for_its_frames_is_refused_and_counted(self) -> None:
        # CTC cannot align more characters than frames, so the loss would be infinite and
        # would poison the batch mean. Counted rather than silently dropped.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = read_manifest(_manifest(directory, [("va", 60), ("unreasonablylong", 20)]))
            alphabet = alphabet_of(samples)

            dataset = SyllableCrops(samples, alphabet, frames_for=lambda width: width // 4)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.too_long, 1)

    def test_a_missing_crop_is_counted_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = read_manifest(_manifest(directory, [("va", 60)]))
            samples.append(Sample(directory / "absent.png", "xx", "s1"))
            alphabet = Alphabet("vax")

            dataset = SyllableCrops(samples, alphabet, frames_for=lambda width: 99)

        self.assertEqual(dataset.unreadable, 1)

    def test_a_crop_comes_back_at_the_model_height(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = read_manifest(_manifest(directory, [("va", 60)]))
            dataset = SyllableCrops(samples, alphabet_of(samples), frames_for=lambda w: 99)

            image, target, text = dataset[0]

        self.assertEqual(image.shape[1], IMAGE_HEIGHT)
        self.assertEqual(text, "va")
        self.assertEqual(len(target), 2)


class TestCollate(unittest.TestCase):
    def _batch(self) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        return [
            (torch.zeros(1, IMAGE_HEIGHT, 20), torch.tensor([1, 2]), "va"),
            (torch.zeros(1, IMAGE_HEIGHT, 60), torch.tensor([3, 4, 5, 6]), "gues"),
        ]

    def test_padding_is_paper_not_ink(self) -> None:
        # Zero-padding pads with ink, and the model learns every syllable ends in a bar.
        batch = collate(self._batch())

        self.assertAlmostEqual(batch["images"][0, 0, 0, -1].item(), PAD_VALUE / 255.0, places=5)

    def test_the_true_width_is_carried_not_the_padded_one(self) -> None:
        # CTC given the padded width would hunt for the label inside the padding.
        batch = collate(self._batch())

        self.assertEqual(batch["widths"].tolist(), [20, 60])

    def test_targets_are_concatenated_with_their_lengths(self) -> None:
        batch = collate(self._batch())

        self.assertEqual(batch["targets"].tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(batch["target_lengths"].tolist(), [2, 4])

    def test_every_image_reaches_the_widest_width(self) -> None:
        batch = collate(self._batch())

        self.assertEqual(batch["images"].shape[3], 60)


if __name__ == "__main__":
    unittest.main()
