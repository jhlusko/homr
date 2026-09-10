"""
Loading syllable crops for the recogniser, and the two ways a batch can be wrong.

Crops vary in width - a syllable is one to seven characters - so a batch has to be padded,
and padding is where a CTC recogniser quietly breaks. Two rules follow from that:

  * Pad on the right with white, the page's own background. Padding with zeros pads with
    *ink*, and the model learns that every syllable ends in a black bar.
  * Tell CTC the true frame count per item, not the padded one. Given the padded width it
    would look for the label somewhere in the padding and find blanks.

The third rule is about labels rather than images. CTC cannot align a label longer than the
frame count, so such an example is not hard, it is impossible - it contributes an infinite
loss that poisons the batch mean. Those are refused at load time with a count, rather than
silently dropped, so a corpus that starts producing them says so.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.architecture.ocr.crnn import IMAGE_HEIGHT, Alphabet

#: The page's background. Crops are grayscale with ink dark on light paper.
PAD_VALUE = 255


@dataclass(frozen=True)
class Sample:
    image: Path
    text: str
    score: str


def read_manifest(path: Path) -> list[Sample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(Sample(Path(row["image"]), row["text"], row.get("score", "")))
    return samples


def alphabet_of(samples: list[Sample]) -> Alphabet:
    """Every character the corpus actually uses.

    Taken from the data rather than declared: 27.42 counted 104 characters across French
    and German typography, and a hand-written list would drop the rare ones - which are
    exactly the ones a model has least chance of guessing.
    """
    return Alphabet("".join(sample.text for sample in samples))


def scaled_width(width: int, height: int, target: int = IMAGE_HEIGHT) -> int:
    """Width after scaling to `target` height, keeping the aspect ratio."""
    return max(8, int(round(width * target / max(1, height))))


class SyllableCrops(Dataset):
    def __init__(
        self, samples: list[Sample], alphabet: Alphabet, frames_for, height: int = IMAGE_HEIGHT
    ) -> None:
        self.alphabet = alphabet
        self.height = height
        self.samples: list[Sample] = []
        self.too_long = 0
        self.unreadable = 0

        for sample in samples:
            image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.unreadable += 1
                continue
            width = scaled_width(image.shape[1], image.shape[0], height)
            if frames_for(width) < len(sample.text):
                # CTC has nowhere to put the characters, so the loss would be infinite.
                self.too_long += 1
                continue
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(
            image, (scaled_width(image.shape[1], image.shape[0], self.height), self.height),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, torch.tensor(self.alphabet.encode(sample.text)), sample.text


def collate(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> dict:
    """Pad to the widest crop, on the right, with paper rather than ink."""
    widest = max(item[0].shape[2] for item in batch)
    height = batch[0][0].shape[1]
    images = torch.full((len(batch), 1, height, widest), PAD_VALUE / 255.0)
    widths = []
    for index, (image, _, _) in enumerate(batch):
        images[index, :, :, : image.shape[2]] = image
        widths.append(image.shape[2])

    return {
        "images": images,
        "targets": torch.cat([item[1] for item in batch]),
        "target_lengths": torch.tensor([len(item[1]) for item in batch]),
        # The unpadded widths. Handing CTC the padded width would have it hunt for the
        # label inside the padding.
        "widths": torch.tensor(widths),
        "texts": [item[2] for item in batch],
    }
