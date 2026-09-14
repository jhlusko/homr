"""The sidecar must be written in the order the token file holds.

`token_lines_to_str` puts every chord through `sort_token_chords`, which ends
`return [sorted(chord) for chord in chords]`. The sidecar used to be written in the
order the caller passed - source order - and `attach_sidecar` pairs the two by position.
So every chord whose sorted order differed from its source order had its notation
scrambled across its own members.

Measured on the rebuilt Lieder corpus: tie labels impossible on 27.6% of endpoints in
polyphonic staves against 5.7% in monophonic ones. Neither of the two representation
changes made looking for it - recording the voice, then the per-voice onset index -
moved that at all, because both were scrambled by the same reordering.
"""

import tempfile
import unittest
from pathlib import Path

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
)
from homr.transformer.vocabulary import EncodedSymbol, sort_token_chords
from training.omr_datasets.notation_sidecar import attach_sidecar, write_sidecar
from training.transformer.training_vocabulary import read_token_lines, token_lines_to_str


def _note(pitch: str, tie: TieState = TieState.NONE, position: str = "upper") -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm="note_2", pitch=pitch, position=position)
    symbol.notation = NoteNotation(
        beam_levels=(BeamLevelState.NOT_APPLICABLE,) * 4,
        stem=StemDirection.UP,
        slurs=((SlurEvent.NONE, SlurSide.UNSPECIFIED),),
        tie=tie,
    )
    return symbol


def _chord(*members: EncodedSymbol) -> list[EncodedSymbol]:
    out: list[EncodedSymbol] = []
    for index, member in enumerate(members):
        if index:
            out.append(EncodedSymbol("chord"))
        out.append(member)
    return out


class TestTheWriterMatchesTheTokenFile(unittest.TestCase):
    def _round_trip(self, symbols: list[EncodedSymbol]) -> list[tuple[str, str]]:
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text(token_lines_to_str(symbols), encoding="utf-8")
            self.assertIsNotNone(write_sidecar(tokens, symbols))
            read_back = read_token_lines(tokens.read_text().splitlines())
            attach_sidecar(tokens, read_back)
            return [
                (s.pitch, str(s.notation.tie))
                for s in read_back
                if s.rhythm.startswith("note") and s.notation is not None
            ]

    def test_a_tie_stays_on_its_own_notehead(self) -> None:
        """The chord sorts to F4, E4 while the source gives E4, F4 - and F4 is tied."""
        got = self._round_trip(_chord(_note("E4"), _note("F4", TieState.START)))
        self.assertEqual(("F4", "start"), next(row for row in got if row[0] == "F4"))
        self.assertEqual(("E4", "none"), next(row for row in got if row[0] == "E4"))

    def test_the_order_written_is_the_order_read(self) -> None:
        symbols = _chord(_note("C4"), _note("G4", TieState.START), _note("E4"))
        expected = [
            s.pitch
            for chord in sort_token_chords(list(symbols))
            for s in chord
            if s.rhythm.startswith("note")
        ]
        self.assertEqual(expected, [pitch for pitch, _tie in self._round_trip(symbols)])

    def test_a_grand_staff_chord_keeps_each_hand_s_own_label(self) -> None:
        symbols = [
            *_chord(_note("G5", TieState.START, "upper")),
            *_chord(_note("E4", TieState.NONE, "lower"), _note("F4", TieState.START, "lower")),
        ]
        got = dict(self._round_trip(symbols))
        self.assertEqual("start", got["G5"])
        self.assertEqual("start", got["F4"])
        self.assertEqual("none", got["E4"])

    def test_a_single_note_is_unaffected(self) -> None:
        self.assertEqual([("G4", "start")], self._round_trip([_note("G4", TieState.START)]))


if __name__ == "__main__":
    unittest.main()
