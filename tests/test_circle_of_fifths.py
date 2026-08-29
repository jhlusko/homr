import os
import unittest
from unittest import mock

from homr.circle_of_fifths import (
    maintain_accidentals_during_measure,
    strip_naturals,
)
from homr.transformer.vocabulary import EncodedSymbol


class TestCircleOfFifths(unittest.TestCase):

    def test_maintain_accidentals_during_measure_with_key_and_barlines(self) -> None:
        symbols = [
            EncodedSymbol("keySignature_1"),
            EncodedSymbol("note_2", "F4", "_"),
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_1", "G4", "_"),
        ]
        result = maintain_accidentals_during_measure(symbols)
        expected = [
            EncodedSymbol("keySignature_1"),
            EncodedSymbol(
                "note_2", "F4", "_"
            ),  # the PrIMus datset encodes the keys already correctly
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("note_4", "G4", "#"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_1", "G4", "_"),
        ]
        self.assertEqual(result, expected)

    def test_maintain_accidentals_during_measure(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F4", "_"),
        ]
        result = maintain_accidentals_during_measure(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F4", "#"),
        ]
        self.assertEqual(result, expected)

    def test_strip_naturals_when_explicitly_disabled(self) -> None:
        symbols = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "N"),
        ]
        with mock.patch.dict(os.environ, {"HOMR_KEEP_NATURALS": "0"}):
            result = strip_naturals(symbols)
        expected = [
            EncodedSymbol("note_4", "F4", "#"),
            EncodedSymbol("note_4", "G4", "_"),
            EncodedSymbol("note_4", "A4", "_"),
            EncodedSymbol("note_4", "F5", "_"),
        ]
        self.assertEqual(result, expected)

    def test_naturals_are_kept_by_default(self) -> None:
        """Flipped default: a matched-control comparison isolated the true PDMX cost at
        -1.2 to -1.6pp (far smaller than the ~5pp a naive comparison suggested, which was
        mostly the known cost of fine-tuning on this corpus at all), against OSSQ N
        recall going 0% -> 62% and OSSQ lift accuracy improving. See strip_naturals'
        docstring."""
        symbols = [EncodedSymbol("note_4", "F5", "N")]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOMR_KEEP_NATURALS", None)
            result = strip_naturals(symbols)
        self.assertEqual(result[0].lift, "N")

    def test_keep_naturals_explicit_1_also_keeps(self) -> None:
        symbols = [EncodedSymbol("note_4", "F5", "N")]
        with mock.patch.dict(os.environ, {"HOMR_KEEP_NATURALS": "1"}):
            result = strip_naturals(symbols)
        self.assertEqual(result[0].lift, "N")
