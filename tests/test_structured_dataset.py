import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

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
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.omr_datasets.notation_sidecar import write_sidecar
from training.transformer.structured_dataset import (
    StructuredNotationDataset,
    target_names,
)
from training.transformer.training_vocabulary import token_lines_to_str


def _notation() -> NoteNotation:
    return NoteNotation(
        beam_levels=(BeamLevelState.BEGIN,) + empty_beam_levels()[1:],
        stem=StemDirection.UP,
        slurs=((SlurEvent.START, SlurSide.ABOVE),) + empty_slur_slots()[1:],
    )


class _Inner:
    """Stands in for the token loader: the dictionary it yields and the paths behind it."""

    def __init__(self, token_path: Path, length: int = 12) -> None:
        self.corpus_list = [{"image": "x.png", "tokens": str(token_path)}]
        self.length = length

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return {
            "inputs": torch.zeros(1, 4, 4),
            "rhythms": torch.zeros(self.length, dtype=torch.long),
            "mask": torch.ones(self.length, dtype=torch.bool),
        }


def _write(tmp: str, annotated: bool) -> Path:
    symbols = [
        EncodedSymbol("clef_G2"),
        EncodedSymbol("note_8", "C5", notation=_notation() if annotated else None),
        EncodedSymbol("barline"),
    ]
    tokens = Path(tmp) / "sample.txt"
    tokens.write_text(token_lines_to_str(symbols), encoding="utf-8")
    if annotated:
        write_sidecar(tokens, symbols)
    return tokens


class TestStructuredNotationDataset(unittest.TestCase):
    def test_the_original_batch_keys_are_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inner = _Inner(_write(tmp, annotated=True))
            item = StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        for key in ("inputs", "rhythms", "mask"):
            self.assertIn(key, item)
        self.assertEqual(item["rhythms"].shape, (12,))

    def test_targets_are_added_under_their_head_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inner = _Inner(_write(tmp, annotated=True))
            item = StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        for name in target_names(2, 1):
            self.assertIn(name, item)
            self.assertEqual(item[name].shape, (12,))

    def test_a_dataset_without_sidecars_yields_no_extra_keys(self) -> None:
        # Corpora that predate the labels must load exactly as before, so the wrapper is
        # safe to leave in place for all of them.
        with tempfile.TemporaryDirectory() as tmp:
            inner = _Inner(_write(tmp, annotated=False))
            item = StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        for name in target_names(2, 1):
            self.assertNotIn(name, item)

    def test_supervision_lands_on_the_note_and_nowhere_else(self) -> None:
        # BOS at 0, clef at 1, the note at 2. A label one place out would train each head
        # on its neighbour and never raise.
        with tempfile.TemporaryDirectory() as tmp:
            inner = _Inner(_write(tmp, annotated=True))
            item = StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        beams = item["beam.level.1"]
        supervised = (beams != IGNORE_INDEX).nonzero().flatten().tolist()
        self.assertEqual(supervised, [2])

    def test_padding_is_never_supervised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            inner = _Inner(_write(tmp, annotated=True))
            item = StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        self.assertTrue(bool((item["slur.slot.1.event"][5:] == IGNORE_INDEX).all()))


class _TwoEntryInner:
    """Mirrors the real loader's substitution: an unreadable image serves the next entry."""

    def __init__(self, entries: list[dict], served: list[int]) -> None:
        self.corpus_list = entries
        self.served = served

    def __len__(self) -> int:
        return len(self.corpus_list)

    def __getitem__(self, index: int) -> dict:
        self.served.append(index)
        return {
            "inputs": torch.zeros(1, 4, 4),
            "rhythms": torch.zeros(12, dtype=torch.long),
            "mask": torch.ones(12, dtype=torch.bool),
        }


class TestUnreadableImagesStayInStep(unittest.TestCase):
    def test_the_wrapper_follows_the_loaders_substitution(self) -> None:
        # The loader silently serves (index + 1) when an image will not decode. If the
        # wrapper kept attaching the original index's sidecar, one staff's image would
        # carry the previous staff's beams and nothing would raise.
        with tempfile.TemporaryDirectory() as tmp:
            tokens = _write(tmp, annotated=True)
            entries = [
                {"image": str(Path(tmp) / "missing.png"), "tokens": str(tokens)},
                {"image": str(Path(tmp) / "also_missing.png"), "tokens": str(tokens)},
            ]
            served: list[int] = []
            inner = _TwoEntryInner(entries, served)

            StructuredNotationDataset(inner, beam_levels=2, slur_slots=1)[0]

        # Neither image is readable, so it comes back to where it started rather than
        # looping forever.
        self.assertEqual(served, [0])

    def test_a_readable_neighbour_is_the_one_served(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokens = _write(tmp, annotated=True)
            good = Path(tmp) / "good.png"
            cv2.imwrite(str(good), np.zeros((16, 32, 3), dtype=np.uint8))
            entries = [
                {"image": str(Path(tmp) / "missing.png"), "tokens": str(tokens)},
                {"image": str(good), "tokens": str(tokens)},
            ]
            served: list[int] = []

            StructuredNotationDataset(_TwoEntryInner(entries, served), 2, 1)[0]

        self.assertEqual(served, [1])


if __name__ == "__main__":
    unittest.main()
