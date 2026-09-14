"""A tie joins two notations of one pitch, or it is not a tie.

The head predicts each note's state independently, so nothing makes the two ends agree.
On IMSLP183800-sys5-v1 the decode wrote eight `<tie>` elements and none of them paired -
three starts whose pitch never recurs, five stops with no start of that pitch before them -
so a page carrying three ties rendered with zero.
"""

import unittest
from dataclasses import replace

from homr.tie_repair import repair_ties
from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    TieState,
    VoiceClass,
)
from homr.transformer.vocabulary import EncodedSymbol


def _note(
    pitch: str = "G4",
    tie: TieState = TieState.NONE,
    rhythm: str = "note_4",
    position: str = "upper",
) -> EncodedSymbol:
    symbol = EncodedSymbol(rhythm=rhythm, pitch=pitch, position=position)
    symbol.notation = NoteNotation(
        beam_levels=(BeamLevelState.NOT_APPLICABLE,) * 4,
        stem=StemDirection.UP,
        slurs=((SlurEvent.NONE, SlurSide.UNSPECIFIED),),
        tie=tie,
        dynamic=DynamicMark.NONE,
        advance=AdvanceClass.NOT_APPLICABLE,
    )
    return symbol


def _ties(staff) -> list[str]:
    return [str(s.notation.tie) for s in staff if s.notation is not None]


class TestThePartnerIsMarked(unittest.TestCase):
    def test_a_start_whose_pitch_recurs_gets_its_stop(self) -> None:
        staff = [_note("G4", TieState.START), _note("G4")]
        report = repair_ties(staff)
        self.assertEqual(1, report.partner_marked)
        self.assertEqual(["start", "stop"], _ties(staff))

    def test_a_tie_the_head_already_joined_is_left_alone(self) -> None:
        staff = [_note("G4", TieState.START), _note("G4", TieState.STOP)]
        report = repair_ties(staff)
        self.assertEqual(1, report.already_paired)
        self.assertEqual(0, report.changed)
        self.assertEqual(["start", "stop"], _ties(staff))

    def test_a_chain_of_ties_keeps_its_middle(self) -> None:
        staff = [
            _note("G4", TieState.START),
            _note("G4", TieState.START),
            _note("G4"),
        ]
        repair_ties(staff)
        self.assertEqual(["start", "start_and_stop", "stop"], _ties(staff))


class TestWhatCannotBeDrawnIsRemoved(unittest.TestCase):
    def test_a_start_whose_pitch_does_not_recur_is_dropped(self) -> None:
        """The assertion could not have been right: there is nothing to tie to."""
        staff = [_note("G4", TieState.START), _note("A4"), _note("B4")]
        report = repair_ties(staff)
        self.assertEqual(1, report.dropped_starts)
        self.assertEqual(["none", "none", "none"], _ties(staff))

    def test_an_orphan_stop_is_dropped(self) -> None:
        staff = [_note("C4"), _note("G4", TieState.STOP)]
        report = repair_ties(staff)
        self.assertEqual(1, report.dropped_stops)
        self.assertEqual(["none", "none"], _ties(staff))

    def test_nothing_ties_across_a_rest(self) -> None:
        """A rest ends the sounding note, so nothing can be tied over it."""
        staff = [_note("G4", TieState.START), _note(".", rhythm="rest_4"), _note("G4")]
        report = repair_ties(staff)
        self.assertEqual(1, report.dropped_starts)
        self.assertEqual("none", _ties(staff)[0])


class TestWhatItRefusesToInvent(unittest.TestCase):
    def test_a_repeated_pitch_is_not_tied_on_its_own(self) -> None:
        """Only one adjacent same-pitch pair in six is tied (15.9%), so the constraint
        is necessary and nowhere near sufficient. Whether a tie exists stays the head's."""
        staff = [_note("G4"), _note("G4"), _note("G4")]
        report = repair_ties(staff)
        self.assertEqual(0, report.changed)
        self.assertEqual(["none", "none", "none"], _ties(staff))

    def test_a_tie_continuing_past_the_voice_is_kept(self) -> None:
        """Ordinary engraving: the partner is in the next system, not in this voice."""
        staff = [_note("C4"), _note("G4", TieState.START)]
        report = repair_ties(staff)
        self.assertEqual(1, report.at_edge)
        self.assertEqual(["none", "start"], _ties(staff))

    def test_a_stop_on_the_first_note_is_kept(self) -> None:
        staff = [_note("G4", TieState.STOP), _note("C4")]
        report = repair_ties(staff)
        self.assertEqual(1, report.at_edge)
        self.assertEqual(["stop", "none"], _ties(staff))


class TestChordsAndStaves(unittest.TestCase):
    def test_a_chord_member_is_not_its_own_partner(self) -> None:
        """A chord's members are consecutive; the partner is in the NEXT chord."""
        staff = [
            _note("G4", TieState.START),
            EncodedSymbol(rhythm="chord"),
            _note("G4"),
        ]
        report = repair_ties(staff)
        self.assertEqual(1, report.dropped_starts)

    def test_a_partner_on_the_other_staff_does_not_count(self) -> None:
        staff = [_note("G4", TieState.START, position="upper"), _note("G4", position="lower")]
        report = repair_ties(staff)
        self.assertEqual(0, report.partner_marked)

    def test_a_grand_staff_ties_within_each_staff(self) -> None:
        staff = [
            _note("G4", TieState.START, position="upper"),
            _note("C3", TieState.START, position="lower"),
            _note("G4", position="upper"),
            _note("C3", position="lower"),
        ]
        repair_ties(staff)
        self.assertEqual(["start", "start", "stop", "stop"], _ties(staff))


class TestNothingToDo(unittest.TestCase):
    def test_a_staff_with_no_ties_is_untouched(self) -> None:
        staff = [_note("G4"), _note("A4")]
        self.assertEqual(0, repair_ties(staff).changed)

    def test_a_staff_with_no_notation_reports_nothing(self) -> None:
        self.assertEqual(0, repair_ties([EncodedSymbol(rhythm="barline")]).changed)


if __name__ == "__main__":
    unittest.main()


class TestVoices(unittest.TestCase):
    """A tie joins one pitch within one voice.

    Two voices sharing a staff were indistinguishable before `VoiceClass` existed, and
    then the next chord on the staff is as likely to be the other voice's. Measured on
    the Lieder corpus: tie labels impossible on 8.8% of endpoints in polyphonic staves
    against 0.0% in monophonic ones.
    """

    @staticmethod
    def _voiced(pitch: str, tie: TieState, voice: VoiceClass) -> EncodedSymbol:
        symbol = _note(pitch, tie)
        symbol.notation = replace(symbol.notation, voice=voice)
        return symbol

    def test_the_other_voice_is_not_a_partner(self) -> None:
        """The upper voice's next note must not close the lower voice's tie."""
        staff = [
            self._voiced("G4", TieState.START, VoiceClass.SECOND),
            self._voiced("B5", TieState.NONE, VoiceClass.FIRST),
            self._voiced("G4", TieState.NONE, VoiceClass.SECOND),
        ]
        report = repair_ties(staff)
        self.assertEqual(1, report.partner_marked)
        self.assertEqual(["start", "none", "stop"], _ties(staff))

    def test_a_tie_is_not_invented_across_voices(self) -> None:
        staff = [
            self._voiced("G4", TieState.START, VoiceClass.FIRST),
            self._voiced("G4", TieState.NONE, VoiceClass.SECOND),
            self._voiced("A4", TieState.NONE, VoiceClass.FIRST),
        ]
        report = repair_ties(staff)
        self.assertEqual(1, report.dropped_starts)

    def test_an_unstated_voice_falls_back_to_the_staff(self) -> None:
        """Every corpus written before voices existed carries UNKNOWN, and must still
        behave exactly as it did - the staff is then the finest division available."""
        staff = [_note("G4", TieState.START), _note("G4")]
        self.assertEqual(1, repair_ties(staff).partner_marked)


class TestOnsetIndexDecidesAdjacency(unittest.TestCase):
    """The chord grouping in the token stream is a simultaneity across every voice.

    Filtering candidates by voice does not fix that, because the grouping itself is
    cross-voice: a voice's successive notes can share a line, and notes on one line can
    belong to three different voices.
    """

    @staticmethod
    def _at(pitch: str, tie: TieState, voice: VoiceClass, onset: int) -> EncodedSymbol:
        symbol = _note(pitch, tie)
        symbol.notation = replace(symbol.notation, voice=voice, onset_index=onset)
        return symbol

    def test_a_partner_in_the_same_line_is_found_when_its_onset_is_later(self) -> None:
        """Two voices share the line, so `chord` cannot separate them - the index can."""
        staff = [
            self._at("G4", TieState.START, VoiceClass.FIRST, 1),
            self._at("C3", TieState.NONE, VoiceClass.SECOND, 1),
            self._at("G4", TieState.NONE, VoiceClass.FIRST, 2),
        ]
        report = repair_ties(staff)
        self.assertEqual(1, report.partner_marked)
        self.assertEqual(["start", "none", "stop"], _ties(staff))

    def test_a_chord_sibling_sharing_an_onset_is_never_the_partner(self) -> None:
        staff = [
            self._at("G4", TieState.START, VoiceClass.FIRST, 1),
            self._at("G4", TieState.NONE, VoiceClass.FIRST, 1),
        ]
        self.assertEqual(1, repair_ties(staff).dropped_starts)

    def test_notes_several_lines_apart_still_pair(self) -> None:
        """Other voices in between must not end the search."""
        staff = [
            self._at("G4", TieState.START, VoiceClass.FIRST, 1),
            self._at("C3", TieState.NONE, VoiceClass.SECOND, 1),
            self._at("D3", TieState.NONE, VoiceClass.SECOND, 2),
            self._at("G4", TieState.NONE, VoiceClass.FIRST, 2),
        ]
        self.assertEqual(1, repair_ties(staff).partner_marked)

    def test_without_indices_the_chord_grouping_is_still_used(self) -> None:
        """Every corpus before schema v6 has none, and must behave as it did."""
        staff = [_note("G4", TieState.START), _note("G4")]
        self.assertEqual(1, repair_ties(staff).partner_marked)
