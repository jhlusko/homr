"""
A CRNN for reading one syllable, and the alphabet it reads into.

27.45 chose CTC over the stronger scene-text recognisers for a reason that is easy to lose:
the targets are fragments. `ter`, `nel`, `Ê`, `va`, `schaft!` are not words, and a decoder
carrying a language-model prior will quietly repair them into words - which is the one
failure mode that would look like good text and be wrong. CTC has no such prior.

The second reason is that CTC's alignment is not a by-product here. Each output frame
corresponds to a horizontal slice of the crop, so the frame where a character peaks gives
its x-position, and the resolve stage that attaches syllables to notes wants exactly that.

Shape: a convolutional stack reduces height to 1 and width by 4, a bidirectional GRU reads
the resulting sequence, and a linear layer emits alphabet+1 logits per frame. Small on
purpose - 27.42 measured a 104-character alphabet and syllables of median 3 characters, so
capacity is not the binding constraint and a large pretrained model would bring back the
language prior this design is avoiding.
"""

import torch
import torch.nn as nn

#: Every crop is scaled to this height, keeping its aspect ratio. 27.47 sampled the render
#: resolution across the range the scans span, so crops arrive between 28 and 78 pixels
#: tall; the model sees one height and the variation survives as the blur and stroke weight
#: that rescaling leaves behind, which is the part worth learning from.
IMAGE_HEIGHT = 32

#: CTC's blank. Index 0 by convention, so a character's own index is never 0 and an
#: all-blank decode is unambiguous.
BLANK = 0


class Alphabet:
    """The characters the recogniser can emit, and the mapping to and from CTC indices.

    Built from the corpus rather than declared, because 27.42's measured alphabet is 104
    characters of French and German typography - accented vowels, guillemets, three kinds of
    dash - and a hand-written list would silently drop the rare ones. Unknown characters at
    encode time raise rather than being skipped: a label the model cannot represent is a
    corpus problem, and skipping it would train the model to omit that character.
    """

    def __init__(self, characters: str) -> None:
        self.characters = "".join(sorted(set(characters)))
        self._to_index = {character: index + 1 for index, character in enumerate(self.characters)}
        self._to_character = {index: character for character, index in self._to_index.items()}

    def __len__(self) -> int:
        """Alphabet size plus the blank, which is what the output layer needs."""
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self._to_index))
        if missing:
            raise KeyError(f"characters outside the alphabet: {missing}")
        return [self._to_index[character] for character in text]

    def decode(self, indices: list[int]) -> str:
        """Collapse repeats and drop blanks - the standard CTC reading."""
        text = []
        previous = None
        for index in indices:
            if index != previous and index != BLANK:
                text.append(self._to_character.get(index, ""))
            previous = index
        return "".join(text)


class CRNN(nn.Module):
    def __init__(self, alphabet_size: int, channels: int = 1, hidden: int = 192) -> None:
        super().__init__()
        # Height is halved four times, 32 -> 2, then a final 2-high kernel flattens it.
        # Width is halved only twice, so a 3-character syllable still gets several frames
        # per character and CTC has room to place them.
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.recurrent = nn.GRU(
            128 * 2, hidden, num_layers=2, bidirectional=True, batch_first=True
        )
        self.classify = nn.Linear(hidden * 2, alphabet_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities shaped (time, batch, alphabet), which is CTC's order."""
        maps = self.features(images)
        batch, channels, height, width = maps.shape
        # (B, C, H, W) -> (B, W, C*H): one vector per horizontal slice, left to right.
        sequence = maps.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
        hidden, _ = self.recurrent(sequence)
        return self.classify(hidden).log_softmax(dim=-1).permute(1, 0, 2)

    def frame_count(self, image_width: int) -> int:
        """How many output frames a crop of this width produces.

        CTC needs the target no longer than the frame count, and a syllable whose label
        cannot fit is unlearnable rather than merely hard - so the caller has to be able to
        ask before training on it.
        """
        return max(1, image_width // 4)
