"""
A small closed-set classifier for Dynamic marks - not the CTC recogniser.

27.94 decided dynamics need this rather than open-vocabulary text reading: `p`/`f`/`mf`/...
is a discrete choice among about a dozen marks (`dynamics_crops.py` found 18 raw labels in
practice, dominated by 8 that cover ~97% of examples), not a variable-length string. A
single softmax over a fixed label set, not a recurrent sequence model - `CRNN`
(`crnn.py`) is the wrong tool here on purpose, not a smaller version of it.

**Crops are resized to a fixed square, not height-normalised like syllable crops.**
`CRNN` keeps width free because CTC reads a strip of any length; there is no sequence here
for a variable width to serve, and Dynamic marks are close to square already (27.45 measured
a 26x26 median at 150dpi) - a fixed size keeps the model and the data pipeline simple, at
the cost of the aspect-ratio distortion a non-square mark takes on. That trade is judged by
the classifier's own accuracy, not assumed acceptable.
"""

import torch
from torch import nn

#: Fixed crop size fed to the network - see module docstring for why square, not
#: height-normalised-width-free like the syllable recogniser.
IMAGE_SIZE = 48


class Labels:
    """The closed set of dynamics labels the classifier can predict, and the mapping to
    and from class indices - built from the corpus rather than declared, same reasoning
    as `crnn.Alphabet`: a hand-written list would silently drop a rare real label.
    """

    def __init__(self, labels: list[str]) -> None:
        self.labels = sorted(set(labels))
        self._to_index = {label: index for index, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, label: str) -> int:
        if label not in self._to_index:
            raise KeyError(f"label outside the trained set: {label!r}")
        return self._to_index[label]

    def decode(self, index: int) -> str:
        return self.labels[index]


class DynamicsCNN(nn.Module):
    def __init__(self, num_classes: int, channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Adaptive pooling absorbs any residual size drift after fixed-size resizing
        # (e.g. an off-by-one from rounding), rather than requiring an exact input shape.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classify = nn.Linear(64, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        maps = self.features(images)
        pooled = self.pool(maps).flatten(1)
        return self.classify(pooled)
