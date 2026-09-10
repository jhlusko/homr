"""
Loading Dynamic crops for the classifier - see `dynamics_crops.py` for how they were cut.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from training.architecture.ocr.dynamics_classifier import IMAGE_SIZE, Labels


@dataclass(frozen=True)
class Sample:
    image: Path
    label: str
    score: str


def read_manifest(path: Path) -> list[Sample]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        samples.append(Sample(Path(row["image"]), row["label"], row.get("score", "")))
    return samples


def labels_of(samples: list[Sample]) -> Labels:
    return Labels([sample.label for sample in samples])


class DynamicsCrops(Dataset):
    def __init__(self, samples: list[Sample], labels: Labels, size: int = IMAGE_SIZE) -> None:
        self.labels = labels
        self.size = size
        # A label absent from training cannot be predicted or scored meaningfully - same
        # reasoning as train_recognizer.py's unrepresentable-character filter, applied to
        # whole labels instead of characters.
        self.samples = [s for s in samples if s.label in labels._to_index]
        self.skipped = len(samples) - len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample.image)
        resized = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, self.labels.encode(sample.label)


def collate(batch: list[tuple[torch.Tensor, int]]) -> dict:
    images = torch.stack([item[0] for item in batch])
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return {"images": images, "targets": targets}
