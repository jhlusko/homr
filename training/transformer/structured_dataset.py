"""
Adding notation targets to a training batch, without disturbing the batch itself.

The existing loader yields the six token branches and an image, and every training path
depends on that shape. This wraps it rather than editing it: the same keys come out
unchanged, with the structured targets added under their head names. A dataset whose
token files have no notation sidecar yields no extra keys at all, so the wrapper is safe
to leave in place for corpora that predate the labels.

The one thing that has to be right is position. Targets are built through
`notation_positions`, which mirrors the decoder's own BOS/symbols/EOS/padding layout, so
a label cannot end up describing its neighbour.
"""

from typing import Any

from torch.utils.data import Dataset

from homr.transformer.structured_notation import NoteNotation
from training.architecture.transformer.structured_targets import (
    build_targets,
    notation_positions,
)
from training.omr_datasets.notation_sidecar import attach_sidecar
from training.transformer.training_vocabulary import read_tokens


class StructuredNotationDataset(Dataset):
    """Wraps a token dataset so each item also carries its notation targets.

    `inner` is any dataset whose items are the loader's dictionary and whose
    `corpus_list[i]["tokens"]` names the token file that item came from - which is what
    lets the sidecar be found without re-deriving the path.
    """

    def __init__(self, inner: Any, beam_levels: int, slur_slots: int) -> None:
        self.inner = inner
        self.beam_levels = beam_levels
        self.slur_slots = slur_slots

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.inner[index])
        token_path = self.inner.corpus_list[index]["tokens"]
        length = int(item["rhythms"].shape[-1])
        positions = self._positions(token_path, length)
        if positions is None:
            return item
        targets = build_targets([positions], self.beam_levels, self.slur_slots)
        for name, tensor in targets.items():
            item[name] = tensor[0]
        return item

    def _positions(self, token_path: str, length: int) -> list[NoteNotation | None] | None:
        """Notation per decoder position, or None when this example has no labels."""
        symbols = read_tokens(token_path)
        if attach_sidecar(token_path, symbols) == 0:
            return None
        return notation_positions(symbols, length)


def target_names(beam_levels: int, slur_slots: int) -> list[str]:
    """The keys StructuredNotationDataset adds, for a collate function to expect."""
    return list(build_targets([[None]], beam_levels, slur_slots))
