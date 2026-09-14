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


class TestTheCompleteSerializationOrder(unittest.TestCase):
    """One simultaneity holding both staves and mixed rhythms, end to end.

    The four tests above all pass on a writer that applies only the FIRST of the token
    writer's two sorts, which is why they did not catch that partial fix. One of them
    derives its expectation from that same first sort, and the grand-staff one puts the
    hands in *separate* simultaneities, so neither can see a cross-staff permutation.

    Measured against raw source tie states over all 4,187 crops
    (docs/TIE_LABEL_FINDINGS.md): 91.8% writing in source order, 80.0% with one sort,
    100% with both.
    """

    def _attached(self, symbols: list[EncodedSymbol]) -> list[tuple[str, str, str]]:
        """(pitch, position, tie) as a reader recovers it from the written files."""
        with tempfile.TemporaryDirectory() as directory:
            tokens = Path(directory) / "x.tokens"
            tokens.write_text(token_lines_to_str(symbols), encoding="utf-8")
            self.assertIsNotNone(write_sidecar(tokens, symbols))
            read_back = read_token_lines(tokens.read_text().splitlines())
            attach_sidecar(tokens, read_back)
            return [
                (s.pitch, s.position, str(s.notation.tie))
                for s in read_back
                if s.rhythm.startswith("note") and s.notation is not None
            ]

    def test_one_simultaneity_across_both_staves_keeps_every_label(self) -> None:
        """The shape that broke: an eighth and a half note in the upper staff and a half
        note in the lower, all sounding together, with the tie on the last of them."""
        symbols = _chord(
            _note("A4", TieState.NONE, "upper"),
            _note("D3", TieState.NONE, "lower"),
            _note("C5", TieState.START, "upper"),
        )
        symbols[0].rhythm = "note_8"
        attached = dict(
            ((pitch, position), tie) for pitch, position, tie in self._attached(symbols)
        )
        self.assertEqual("start", attached[("C5", "upper")])
        self.assertEqual("none", attached[("A4", "upper")])
        self.assertEqual("none", attached[("D3", "lower")])

    def test_every_member_keeps_its_own_identity(self) -> None:
        """Each notehead gets a distinguishable label, so any permutation shows up."""
        states = (TieState.START, TieState.STOP, TieState.NONE, TieState.START_AND_STOP)
        pitches = ("C4", "G4", "E5", "A3")
        positions = ("lower", "upper", "upper", "lower")
        symbols = _chord(
            *(
                _note(pitch, state, position)
                for pitch, state, position in zip(pitches, states, positions, strict=True)
            )
        )
        expected = {pitch: str(state) for pitch, state in zip(pitches, states, strict=True)}
        self.assertEqual(expected, {pitch: tie for pitch, _pos, tie in self._attached(symbols)})

    def test_the_writers_agree_about_the_order(self) -> None:
        """Not an assertion about which order - an assertion that there is only one."""
        symbols = _chord(
            _note("C4", TieState.NONE, "lower"),
            _note("G4", TieState.START, "upper"),
            _note("E5", TieState.STOP, "upper"),
        )
        from_tokens = [
            line.split()[1]
            for line in token_lines_to_str(symbols).replace("&", "\n").splitlines()
            if line.split()[0].startswith("note")
        ]
        self.assertEqual(from_tokens, [pitch for pitch, _pos, _tie in self._attached(symbols)])
